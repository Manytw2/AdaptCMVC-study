import logging
import math
import argparse
import os
# 在导入 torch 之前禁用 CUDA，避免在 CPU 模式下加载 CUDA DLL（节省内存）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 默认禁用 GPU
os.environ['TORCH_CUDA_ARCH_LIST'] = ''  # 禁用 CUDA 架构检测
import torch
import torch.optim as optim
from optimizer import get_optimizer
from robustbench.data import load_multiview
from robustbench.utils import clean_accuracy_source as accuracy_source
from collections import defaultdict

from robustbench.base_model import ConsistencyAE

import cotta
from data_load import  get_val_transformations,get_train_dataset
import numpy as np
logger = logging.getLogger(__name__)


def train_a_epoch(x, y, model, epoch, optimizer, device):
    x_test, y_test = x[0].to(device), y.to(device)
    losses = defaultdict(list)


    model.train()
    parameters = list(model.parameters())

    mask_ratio, mask_patch_size = 0.7, 4

    n_batches = math.ceil(x_test.shape[0] / args.BATCH_SIZE)



    for counter in range(n_batches):
        x_curr = x_test[counter * args.BATCH_SIZE:(counter + 1) *
                                                  args.BATCH_SIZE]


        y_curr = y_test[counter * args.BATCH_SIZE:(counter + 1) *
                                            args.BATCH_SIZE]


        loss, loss_parts = model.get_loss([x_curr], epoch, mask_ratio, mask_patch_size)

        for k, v in loss_parts.items():
            losses[k].append(v)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        show_losses = {k: np.mean(v) for k, v in losses.items()}

    result,kmeans_pre, kmeans_center,out_reprs = accuracy_source(model, x_test, y_test, args.BATCH_SIZE, args.class_num, device=device)
    print(f"[Evaluation]", ', '.join([f'{k}:{v:.4f}' for k, v in result.items()]))

    return show_losses, loss.item(), result['consist-acc'], kmeans_pre, kmeans_center, out_reprs


def evaluate(description, device):
    AE = ConsistencyAE(basic_hidden_dim=32, c_dim=20, continous=True, in_channel=3, num_res_blocks=3,
                       ch_mult=[1, 2, 4, 8],
                       block_size=8, temperature=1.0,  ###
                       latent_ch=8, kld_weight=1.0, views=1, categorical_dim=args.class_num)
    base_model = AE

    base_model = base_model.to(device)


    if args.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)

    if args.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)


    val_transformations = get_val_transformations(args.CROP_SIZE)
    train_dataset = get_train_dataset(args.name,args.root,args.views, val_transformations)


    views = 0


    optimizer = get_optimizer(model.parameters(), args.LR, args.METHOD)
    x_test, y_test = load_multiview(args.NUM_EX, False, train_dataset)
    result_dir = os.path.join("./source")
    best_loss = np.inf
    best_acc = 0.
    old_best_model_path = ""
    acc_list = []
    loss_list = []
    for epoch in range(200):



        losses, cur_loss, cur_acc, kmeans_pre, kmeans_center, out_reprs = train_a_epoch(x_test, y_test, model, epoch, optimizer, device)
        loss_list.append(cur_loss)
        acc_list.append(cur_acc)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_model_path = os.path.join(result_dir, f'20dim-best-{int(best_loss)}-{epoch}-{cur_acc}.pth')
            if old_best_model_path:
                # save storage.
                os.remove(old_best_model_path)
            old_best_model_path = best_model_path
            np.save(f'./source/source_result.npy', kmeans_pre)
            np.save(f'./source/source_center.npy', kmeans_center)
            np.save(f'./source/source_feature.npy', out_reprs)


            model.eval()
                # Save final model
            torch.save(model.state_dict(), best_model_path)
    np.save(f'./source/loss-{int(views)}.npy', np.array(loss_list))
    np.save(f'./source/acc-{int(views)}.npy', np.array(acc_list))





def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    # logger.info(f"model for evaluation: %s", model)
    return model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
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
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=args.STEPS,
                           episodic=args.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model


import random

if __name__ == '__main__':


    params = {'alpha':1.0,'BATCH_SIZE':64}
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=params['alpha'], help='alpha')#1.0
    parser.add_argument('--cuda_device', type=str, default='cpu', help='The number of cuda device. Use "cpu" for CPU mode.')
    parser.add_argument('--seed', type=str, default=3407, help='seed')
    parser.add_argument('--CROP_SIZE', type=int, default=64, help='CROP_SIZE')  # 50
    parser.add_argument('--ADAPTATION', type=str, default='source', help='direction of datasets')
    parser.add_argument('--name', type=str, default='coil-100', help='name')
    parser.add_argument('--root', type=str, default='MyData', help='root')
    parser.add_argument('--views', type=int, default=3, help='views')
    parser.add_argument('--NUM_EX', type=int, default=1920, help='NUM_EX')
    parser.add_argument('--BATCH_SIZE', type=int, default=params['BATCH_SIZE'], help='BATCH_SIZE') #256
    parser.add_argument('--class_num', type=int, default=100, help='class_num')
    parser.add_argument('--STEPS', type=int, default=1, help='class_num')
    parser.add_argument('--EPISODIC', action='store_true', default=False, help='EPISODIC')
    parser.add_argument('--LR', type=float, default=0.0001, help='LR')
    parser.add_argument('--BETA', type=float, default=0.9, help='BETA')
    parser.add_argument('--WD', type=float, default=0.0, help='WD')
    parser.add_argument('--METHOD', type=str, default='adamw', help='METHOD')


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
    evaluate('"Imagenet-C evaluation.', device)
