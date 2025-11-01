import logging
import numpy as np
import argparse
import os
# 在导入 torch 之前禁用 CUDA，避免在 CPU 模式下加载 CUDA DLL（节省内存）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 默认禁用 GPU
os.environ['TORCH_CUDA_ARCH_LIST'] = ''  # 禁用 CUDA 架构检测
import torch
import torch.optim as optim
from robustbench.data import load_multiview
from robustbench.sim_utils import clean_accuracy_target as accuracy_target
from robustbench.base_model import ConsistencyAE

import cotta

from data_load import  get_val_transformations,get_train_dataset

logger = logging.getLogger(__name__)



def evaluate(device):
    """
    运行一次增量视图适配与评估流程（AdaptCMVC 主流程）。

    流程概览（与 think.md 对齐）：
    - 构建基础一致性自编码器（VAE 变体）作为学生模型，后续由 CoTTA 包装形成教师-学生框架（EMA 教师）。
    - 按视角 views=1..V-1 迭代：
      * 加载上一阶段最优权重与聚类先验（标签与中心）。
      * 采用自训练的噪声鲁棒一致性损失：教师-学生一致性，且以"到历史原型的距离"作权重（距离越小权重越大）。
      * 采用结构对齐损失：对齐当前特征的相似度矩阵与历史共识结构矩阵 S（逐步 EMA 更新）。
      * 记录并选择最优模型（以重构与一致性损失之和作为准则），保存权重、聚类结果与中心。
    - 返回最后一个视角的最佳一致性聚类精度（consist-acc）。
    """

    AE = ConsistencyAE(basic_hidden_dim=32, c_dim=20, continous=True, in_channel=3, num_res_blocks=3,
                       ch_mult=[1, 2, 4, 8],
                       block_size=8, temperature=1.0,  ###
                       latent_ch=8, kld_weight=1.0, views=1, categorical_dim=args.class_num)


    val_transformations = get_val_transformations(args.CROP_SIZE)
    train_dataset = get_train_dataset(args.name,args.root,args.views, val_transformations)

    for views in range(1, args.views):
        print('###############################views:',views,'######################################')
        if views == 1:
            # 第一个增量视角：加载源模型
            AE.load_state_dict(torch.load('./source_model/v1-best-2806-185-0.5641.pth', map_location='cpu'),
                           strict=False)
        else:
            # 后续视角：加载上一视角的模型
            AE.load_state_dict(torch.load(f'./last_sim_model/last_sim_model--{int(views-1)}.pth', map_location='cpu'),
                               strict=False)


        base_model = AE
        base_model = base_model.to(device)

        if args.ADAPTATION == "source":
            logger.info("test-time adaptation: NONE")
            model = setup_source(base_model)
        if args.ADAPTATION == "cotta":
            logger.info("test-time adaptation: CoTTA")
            model = setup_cotta(base_model)


        best_loss = np.inf
        best_acc = 0.
        result_dir = os.path.join("./last_sim_model")
        os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
        old_best_model_path = ""

        if views <= 1:
            # 第一个增量视角：使用提供的源域先验（历史聚类标签与原型）
            source_result = np.load(f'./source_model/v1-20source_result.npy')  # v1-20source_result.npy
            source_result = torch.from_numpy(source_result).to(device)
            source_center = np.load(f'./source_model/v1-20source_center.npy')  # v1-20source_center.npy
        else:
            # 后续视角：承接上一视角的最优结果，持续更新
            source_result = np.load(f'./last_sim_model/last_sim_result.npy')  # flag_source_result.npy
            source_result = torch.from_numpy(source_result).to(device)
            source_center = np.load(f'./last_sim_model/last_sim_center.npy')  # flag_source_centers.npy


        acc = []
        for i in range(50):
            try:
                if views == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning("not resetting model")
            except:
                logger.warning("not resetting model")

            x_test, y_test = load_multiview(args.NUM_EX,False,train_dataset)
            x_test, y_test = x_test[views].to(device), y_test.to(device)
           

            # accuracy_target 内部执行：
            # - 教师-学生一致性（带距离感知权重）：反映公式 \mathcal{L}_c
            # - 结构对齐损失：当前特征相似度 vs 历史共识结构 S，反映公式 \mathcal{L}_s
            # - 重构/ELBO：反映公式 \mathcal{L}_r
            # 并进行 S 的 EMA 更新（保存为 average_simmatrix.npy）
            result, kmeans_pre, r_loss, c_loss, str_loss, cluster_center = accuracy_target(
                source_center, source_result, model, x_test, y_test,
                args.BATCH_SIZE, args.class_num, views, args.up_alpha, device=device
            )

            cur_loss = r_loss + c_loss
            acc.append(result['consist-acc'])

            print(f"[Evaluation]", ', '.join([f'{k}:{v:.4f}' for k, v in result.items()]))

            if cur_loss <= best_loss:
                best_loss = cur_loss.item()
                best_acc = result['consist-acc']
                best_model_path = os.path.join(result_dir, f'last_sim_model--{int(views)}.pth')
                if old_best_model_path:
                    # save storage.
                    os.remove(old_best_model_path)
                old_best_model_path = best_model_path
                # 保存当前视角的最佳聚类标签与中心（供下一视角承接）
                np.save(f'./last_sim_model/last_sim_result.npy', kmeans_pre)
                np.save(f'./last_sim_model/last_sim_center.npy', cluster_center)


                model.eval()
                # Save final model
                torch.save(model.state_dict(), best_model_path)
        print('best acc',best_acc)
        np.save(f'./last_sim_model/acc-{int(views)}.npy', np.array(acc))
    return best_acc

    



def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    """
    if args.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=args.LR,
                    betas=(args.BETA, 0.999),
                    weight_decay=args.WD)
    elif args.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=args.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=args.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):

    model = cotta.configure_model(model)

    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=args.STEPS,
                           episodic=args.EPISODIC,
                            num_classes = args.class_num,
                              CROP_SIZE = args.CROP_SIZE,
                              contra=args.contra,
                              consis=args.consis,
                              N=args.N,
                              sample_num_each_clusters = args.sample_num_each_clusters)
    return cotta_model



import random

if __name__ == '__main__':


    params = {'alpha': 0.1, 'BATCH_SIZE': 128, 'consis': 5, 'N': 1, 'sample_num_each_clusters': 15, 'contra': 0.5, 'temperature': 0.5, 'up_alpha': 0.2}

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=params['alpha'], help='alpha')  # 1.0
    parser.add_argument('--cuda_device', type=str, default='cpu', help='The number of cuda device. Use "cpu" for CPU mode.')
    parser.add_argument('--seed', type=str, default=3407, help='seed')
    parser.add_argument('--CROP_SIZE', type=int, default=64, help='CROP_SIZE')  # 50
    parser.add_argument('--ADAPTATION', type=str, default='cotta', help='direction of datasets')
    parser.add_argument('--name', type=str, default='coil-100', help='name')
    parser.add_argument('--root', type=str, default='MyData', help='root')
    parser.add_argument('--views', type=int, default=3, help='views')
    parser.add_argument('--NUM_EX', type=int, default=1920, help='NUM_EX')
    parser.add_argument('--BATCH_SIZE', type=int, default=params['BATCH_SIZE'], help='BATCH_SIZE')  # 256
    parser.add_argument('--class_num', type=int, default=100, help='class_num')
    parser.add_argument('--STEPS', type=int, default=1, help='class_num')
    parser.add_argument('--EPISODIC', action='store_true', default=False, help='EPISODIC')
    parser.add_argument('--LR', type=float, default=0.0001, help='LR')
    parser.add_argument('--BETA', type=float, default=0.9, help='BETA')
    parser.add_argument('--WD', type=float, default=0.0, help='WD')
    parser.add_argument('--METHOD', type=str, default='Adam', help='METHOD')
    parser.add_argument('--temperature', type=float, default=params['temperature'], help='temperature')
    parser.add_argument('--contra', type=float, default=params['contra'], help='contra')
    parser.add_argument('--consis', type=float, default=params['consis'], help='consis')
    parser.add_argument('--sample_num_each_clusters', type=int, default=params['sample_num_each_clusters'],
                        help='sample_num_each_clusters')  # 15
    parser.add_argument('--N', type=int, default=params['N'], help='N')  # 3
    parser.add_argument('--up_alpha', type=int, default=params['up_alpha'], help='up_alpha')  # 3






    args = parser.parse_args()
    
    # 设置设备（CPU 或 GPU）
    if args.cuda_device.lower() == 'cpu':
        device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 GPU
    else:
        device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    
    print(f"使用设备: {device}")


    def setup_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if device.type == 'cuda':
            torch.backends.cudnn.deterministic = True

    setup_seed(args.seed)
    best_acc = evaluate(device)

