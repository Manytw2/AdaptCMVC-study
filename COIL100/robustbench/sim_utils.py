import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

# 移除未使用的外部依赖导入（如 requests、gdown），避免环境未安装时报错
import torch
from torch import nn

# 移除对外部 robustbench.model_zoo 的依赖，保留实际使用到的 metric 导入
from robustbench.metric import clustering_by_representation, clustering_accuracy
import numpy as np



def clean_accuracy_source(model: nn.Module,  #######################
                          x: torch.Tensor,
                          y: torch.Tensor,
                          batch_size: int = 100,
                          class_num: int = 10,
                          device: torch.device = None):
    """
    源域（预训练阶段）一致性聚类评估：
    - 取出模型的表示 z，使用 k-means（在 clustering_by_representation 内部）计算 ACC/NMI/ARI/P/F-score。
    - 返回 kmeans 结果（标签与中心），作为后续增量视图的先验。
    - 与 think.md 对齐：这是初始 VAE 阶段的评估输出，为后续自训练提供 B^{v-1} 与历史分配。
    """
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    targets = []
    out_reprs = []

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                                            batch_size].to(device)

            y_curr = y[counter * batch_size:(counter + 1) *
                                            batch_size]

            _, output_z = model([x_curr])  # sourec时ae时model 输出有两个， 目标域时cotta是model 输出只有output

            targets.append(y_curr.detach().cpu())
            out_reprs.append(output_z.detach().cpu())

    targets = torch.concat(targets, dim=-1).numpy()
    out_reprs = torch.vstack(out_reprs).detach().cpu().numpy()

    acc, nmi, ari, _, p, fscore, kmeans_pre, kmeans_center = clustering_by_representation(out_reprs, out_reprs, targets,
                                                                                          class_num)
    # acc += (output.max(1)[1] == y_curr).float().sum()
    result = {}
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    return result, kmeans_pre, kmeans_center

def clu_sim_matrix(last_pre, class_num):
    """
    根据上一阶段的聚类标签 last_pre 构造二值同簇矩阵 S：同类为 1，异类为 0。
    - 与 think.md 的 S 初始化一致（公式 9 的二值化思想）。
    """
    device = last_pre.device if isinstance(last_pre, torch.Tensor) else torch.device('cpu')
    one_hot = torch.zeros(last_pre.shape[0], class_num, device=device)
    one_hot.scatter_(dim=1, index=last_pre.unsqueeze(dim=1).long(),
                     src=torch.ones(last_pre.shape[0], class_num, device=device))
    clu_sim = torch.matmul(one_hot, one_hot.t())

    return clu_sim

def clean_accuracy_target(source_center: np.array,
                          source_result: torch.Tensor,
                          model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          batch_size: int = 100,
                          class_num: int = 10,
                          views: int=1,
                          up_alpha: float=0.1,
                          device: torch.device = None):
    """
    目标域（增量视图）自适应 + 评估：
    - 输入：
      * source_center: 历史聚类原型 B^{v-1}
      * source_result: 历史聚类标签（上一阶段的预测/共识）
      * model: CoTTA 包装后的教师-学生结构（内部含 EMA 教师与 Anchor）
    - 过程：
      * 构造历史结构矩阵 S^{v-1} 或从 average_simmatrix.npy 读取；
      * 前向并计算三项损失：\mathcal{L}_r, \mathcal{L}_c（距离加权一致性）, \mathcal{L}_s（结构对齐）；
      * 聚类评估得到 kmeans_pre 与新的聚类中心；
      * 使用 EMA 方式更新并保存平均结构矩阵 average_simmatrix.npy（对应 think.md 公式 10）。
    - 输出：评估指标、kmeans 结果、各项损失均值、聚类中心。
    """
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    targets = []
    out_reprs = []
    last_pre = []
    r_loss = 0.
    c_loss = 0.

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                                            batch_size].to(device)

            y_curr = y[counter * batch_size:(counter + 1) *
                                            batch_size].to(device)

            y_last = source_result[counter * batch_size:(counter + 1) *
                                                        batch_size].to(device)
            if views <= 1 :
                s_last = clu_sim_matrix(y_last,class_num) #view2
            else:
                sim_matrix = np.load('./average_simmatrix.npy', allow_pickle=True)
                s_last = torch.from_numpy(sim_matrix[counter]).to(device)


            input = ([x_curr], y_last, source_center, s_last)
            output_z, rec_loss, consis_loss, str_loss = model(input)  #

            targets.append(y_curr.detach().cpu())
            out_reprs.append(output_z.detach().cpu())
            last_pre.append(y_last.detach().cpu())
            r_loss += rec_loss
            c_loss += consis_loss

        targets = torch.concat(targets, dim=-1).numpy()
        out_reprs = torch.vstack(out_reprs).detach().cpu().numpy()
        last_pre = torch.concat(last_pre, dim=-1).numpy()

        acc, nmi, ari, _, p, fscore, kmeans_pre, cluster_center = clustering_by_representation(out_reprs,targets, class_num)
        s_update = []
        for counter in range(n_batches):
            pre_last = source_result[counter * batch_size:(counter + 1) *
                                                        batch_size].to(device)
            new_matrix = clu_sim_matrix(torch.from_numpy(kmeans_pre[counter * batch_size:(counter + 1) *batch_size]).to(device),class_num)
            last_matrix = clu_sim_matrix(pre_last,class_num)
            s = up_alpha * new_matrix+ (1 - up_alpha) * last_matrix
            s_update.append(s.cpu().detach().numpy())
        np.save("average_simmatrix.npy", s_update)

        result = {}
        result['consist-acc'] = acc
        result['consist-nmi'] = nmi
        result['consist-ari'] = ari
        result['consist-p'] = p
        result['consist-fscore'] = fscore

        return result, kmeans_pre, r_loss / n_batches, c_loss / n_batches, str_loss/n_batches, cluster_center



