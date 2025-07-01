import logging
import math
import torch
import torch.optim as optim
from optimizer import get_optimizer
import os
from robustbench.data import load_multiview
from robustbench.utils import clean_accuracy_source as accuracy_source
from collections import defaultdict
from robustbench.base_model import ConsistencyAE
from data_load import  get_val_transformations,get_train_dataset
import numpy as np
logger = logging.getLogger(__name__)


def train_a_epoch(x, y, model, epoch, optimizer):
    x_test, y_test = x[0].cuda(), y.cuda()

    losses = defaultdict(list)


    model.train()
    parameters = list(model.parameters())

    mask_ratio, mask_patch_size = 0.7, 4

    n_batches = math.ceil(x_test.shape[0] / args.BATCH_SIZE)



    for counter in range(n_batches):
        x_curr = x_test[counter * args.BATCH_SIZE:(counter + 1) *
                                            args.BATCH_SIZE]
        loss, loss_parts = model.get_loss([x_curr], epoch, mask_ratio, mask_patch_size)

        for k, v in loss_parts.items():
            losses[k].append(v)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        show_losses = {k: np.mean(v) for k, v in losses.items()}

    result,kmeans_pre, kmeans_center, feature = accuracy_source(model, x_test, y_test, args.BATCH_SIZE, args.class_num)



    print(f"[Evaluation]", ', '.join([f'{k}:{v:.4f}' for k, v in result.items()]))

    return show_losses, loss.item(), result['consist-acc'], kmeans_pre, kmeans_center, feature


def evaluate(description):
    AE = ConsistencyAE(basic_hidden_dim=16,c_dim=20,continous=True,in_channel=1,num_res_blocks=3,ch_mult=[1, 2, 4],block_size=8,temperature=1.0,###
                                            latent_ch=8,kld_weight=1.0,categorical_dim=10)

    base_model = AE

    base_model = base_model.cuda()


    if args.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)


    val_transformations = get_val_transformations(args.CROP_SIZE)
    train_dataset = get_train_dataset(args.name,args.root,args.views, val_transformations)

    optimizer = get_optimizer(model.parameters(), args.LR, args.METHOD)
    x_test, y_test = load_multiview(args.NUM_EX, False, train_dataset)
    result_dir = os.path.join("./source")
    best_loss = np.inf
    best_acc = 0.
    old_best_model_path = ""
    acc_list = []
    loss_list = []
    for epoch in range(100):

        losses, cur_loss, cur_acc, kmeans_pre, kmeans_center, feature = train_a_epoch(x_test, y_test, model, epoch, optimizer)
        loss_list.append(cur_loss)
        acc_list.append(cur_acc)

        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_model_path = os.path.join(result_dir, f'source_dim-best-{int(best_loss)}-{epoch}-{cur_acc}.pth')
            if old_best_model_path:
                # save storage.
                os.remove(old_best_model_path)
            old_best_model_path = best_model_path
            np.save(f'./source/source_feature_v1.npy', feature)
            np.save(f'./source/source_kmeans_pre_v1.npy', kmeans_pre)
            np.save(f'./source/source_kmeans_center_v1.npy', kmeans_center)


            model.eval()
                # Save final model
            torch.save(model.state_dict(), best_model_path)


       




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



import argparse
import random
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
    parser.add_argument('--seed', type=str, default=3407, help='seed')
    parser.add_argument('--CROP_SIZE', type=int, default=32, help='CROP_SIZE')  # 50
    parser.add_argument('--ADAPTATION', type=str, default='source', help='direction of datasets')
    parser.add_argument('--name', type=str, default='EdgeMnist', help='name')
    parser.add_argument('--root', type=str, default='MyData', help='root')
    parser.add_argument('--views', type=int, default=2, help='views')
    parser.add_argument('--NUM_EX', type=int, default=60000, help='NUM_EX')
    parser.add_argument('--BATCH_SIZE', type=int, default=256, help='BATCH_SIZE')  # 256
    parser.add_argument('--class_num', type=int, default=10, help='class_num')
    parser.add_argument('--STEPS', type=int, default=1, help='class_num')
    parser.add_argument('--EPISODIC', action='store_true', default=False, help='EPISODIC')
    parser.add_argument('--LR', type=float, default=0.001, help='LR')  # 0.0005
    parser.add_argument('--BETA', type=float, default=0.9, help='BETA')
    parser.add_argument('--WD', type=float, default=0.0, help='WD')  # 0.0
    parser.add_argument('--METHOD', type=str, default='adamw', help='METHOD')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(args.seed)
    evaluate('"Imagenet-C evaluation.')

