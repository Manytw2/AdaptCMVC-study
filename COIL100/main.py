import logging
import numpy as np
import torch
import torch.optim as optim
import argparse
import os
from robustbench.data import load_multiview
from robustbench.sim_utils import clean_accuracy_target as accuracy_target
from robustbench.base_model import ConsistencyAE

import cotta

from data_load import  get_val_transformations,get_train_dataset

logger = logging.getLogger(__name__)



def evaluate():

    AE = ConsistencyAE(basic_hidden_dim=32, c_dim=20, continous=True, in_channel=3, num_res_blocks=3,
                       ch_mult=[1, 2, 4, 8],
                       block_size=8, temperature=1.0,  ###
                       latent_ch=8, kld_weight=1.0, views=1, categorical_dim=args.class_num)


    val_transformations = get_val_transformations(args.CROP_SIZE)
    train_dataset = get_train_dataset(args.name,args.root,args.views, val_transformations)

    for views in range(1, args.views):
        print('###############################views:',views,'######################################')
        if views<=1:
            AE.load_state_dict(torch.load('./source_model/v1-best-2806-185-0.5641.pth', map_location='cpu'),
                           strict=False)
        else:
            AE.load_state_dict(torch.load(f'./last_sim_model/last_sim_model--{int(views)}.pth', map_location='cpu'),
                               strict=False)


        base_model = AE
        base_model = base_model.cuda()

        if args.ADAPTATION == "source":
            logger.info("test-time adaptation: NONE")
            model = setup_source(base_model)
        if args.ADAPTATION == "cotta":
            logger.info("test-time adaptation: CoTTA")
            model = setup_cotta(base_model)


        best_loss = np.inf
        best_acc = 0.
        result_dir = os.path.join("./last_sim_model")
        old_best_model_path = ""

        if views <= 1:
            source_result = np.load(f'./source_model/v1-20source_result.npy')  # v1-20source_result.npy
            source_result = torch.from_numpy(source_result).cuda()
            source_center = np.load(f'./source_model/v1-20source_center.npy')  # v1-20source_center.npy
        else:
            source_result = np.load(f'./last_sim_model/last_sim_result.npy')  # flag_source_result.npy
            source_result = torch.from_numpy(source_result).cuda()
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
            x_test, y_test = x_test[views].cuda(), y_test.cuda()
           

            result, kmeans_pre, r_loss, c_loss, str_loss, cluster_center = accuracy_target(source_center,source_result, model, x_test, y_test, args.BATCH_SIZE, args.class_num,views,args.up_alpha)

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
    parser.add_argument('--cuda_device', type=str, default='0', help='The number of cuda device.')
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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(args.seed)
    best_acc = evaluate()

