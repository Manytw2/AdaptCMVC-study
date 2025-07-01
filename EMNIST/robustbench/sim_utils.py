import math
import torch
from torch import nn

from robustbench.metric import clustering_by_representation
import numpy as np



def clean_accuracy_source(model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          batch_size: int = 100,
                          class_num: int = 10,
                          device: torch.device = None):
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

            _, output_z = model([x_curr])

            targets.append(y_curr.detach().cpu())
            out_reprs.append(output_z.detach().cpu())

    targets = torch.concat(targets, dim=-1).numpy()
    out_reprs = torch.vstack(out_reprs).detach().cpu().numpy()

    acc, nmi, ari, _, p, fscore, kmeans_pre, kmeans_center = clustering_by_representation(out_reprs, out_reprs, targets,
                                                                                          class_num)

    result = {}
    result['consist-acc'] = acc
    result['consist-nmi'] = nmi
    result['consist-ari'] = ari
    result['consist-p'] = p
    result['consist-fscore'] = fscore

    return result, kmeans_pre, kmeans_center

def clu_sim_matrix(last_pre, class_num):

    one_hot = torch.zeros(last_pre.shape[0], class_num).cuda()
    one_hot.scatter_(dim=1, index=last_pre.unsqueeze(dim=1).long(),
                     src=torch.ones(last_pre.shape[0], class_num).cuda())
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
    if device is None:
        device = x.device
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

        acc, nmi, ari, _, p, fscore, kmeans_pre, cluster_center = clustering_by_representation(out_reprs,targets, class_num)
        s_update = []
        for counter in range(n_batches):
            pre_last = source_result[counter * batch_size:(counter + 1) *
                                                         batch_size].to(device)
            new_matrix = clu_sim_matrix(torch.from_numpy(kmeans_pre[counter * batch_size:(counter + 1) *batch_size]).cuda(),class_num)
            last_matrix = clu_sim_matrix(pre_last,class_num)
            s = up_alpha * new_matrix+ (1 - up_alpha) * last_matrix

            s_update.append(s.cpu().detach().numpy())
        s_update = np.asarray(s_update,dtype=object)
        np.save(f'average_simmatrix.npy', s_update)

        result = {}
        result['consist-acc'] = acc
        result['consist-nmi'] = nmi
        result['consist-ari'] = ari
        result['consist-p'] = p
        result['consist-fscore'] = fscore

        return result, kmeans_pre, r_loss / n_batches, c_loss / n_batches, str_loss/n_batches




