from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging
import torch.nn.functional as F


def get_tta_transforms(gaussian_std: float=0.005, soft=True, clip_inputs=False):
   # img_shape = (64, 64, 1) coil20
    img_shape = (32, 32, 1)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
       # transforms.ToTensor(),
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):#, iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

import torch.nn as nn
import torch.nn.functional as F
L2norm = nn.functional.normalize

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda()
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False,num_classes=20,temperature=0.2,contra=1.0,consis=1.0,N=2,sample_num_each_clusters=10):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        #model_anchor参数相同但是可以更新的模型（教师模型） ，   model_ema参数相同但是不能更新的模型    model_state学生模型
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms(0.005, False, False)
        self.num_classes = num_classes
        self.cl = ContrastiveLoss(temperature)
        self.contra = contra
        self.consis = consis
        self.N = N
        # self.alpha_teacher = alpha_teacher
        self.sample_num_each_clusters = sample_num_each_clusters


    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs,rec_loss,consis_loss = self.forward_and_adapt(x, self.optimizer)

        return outputs,rec_loss,consis_loss

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer):
        import numpy as np
        input = x[0]
        last_pre = x[1]
        source_center = x[2]
        recon, z = self.model(input)  # output的维度c_dim目前等于类别数
        _, standard_ema = self.model_ema(input)
        self.model_ema.train()
        # Teacher Prediction
        _, anchor_output = self.model_anchor(input)  # 后面尝试用anchor_output来找top k个距离类中心最近的点

        distances_to_cluster_centers = np.linalg.norm(anchor_output.cpu().detach().numpy() - source_center[last_pre.cpu().detach().numpy()], axis=1)

        def get_top_k_sample_names_for_each_center(k):
            # closest_sample_names = {}
            closest_sample_names = []

            for i in range(self.num_classes):
                cluster_distances = distances_to_cluster_centers[last_pre.cpu().detach().numpy() == i]
                cluster_indices = np.where(last_pre.cpu().detach().numpy() == i)[0]

                top_k_indices = cluster_distances.argsort()[:k]

                # top_k_names = [img_path_list[idx] for idx in cluster_indices[top_k_indices]]
                # # print([distances_to_cluster_centers[idx] for idx in cluster_indices[top_k_indices]])
                # closest_sample_names[i] = top_k_names

                closest_sample_names.extend(cluster_indices[top_k_indices])  #cluster_indices[top_k_indices]

            closest_sample_names = np.sort(closest_sample_names)
            return closest_sample_names

        top_k_samples_name = get_top_k_sample_names_for_each_center(self.sample_num_each_clusters)
        other_samples_name = [i for i in range(input[0].shape[0]) if i not in top_k_samples_name]


        one_hot = torch.zeros(last_pre.shape[0], self.num_classes).cuda()
        one_hot.scatter_(dim=1, index=last_pre.unsqueeze(dim=1).long(),
                         src=torch.ones(last_pre.shape[0], self.num_classes).cuda())
        clu_sim = torch.matmul(one_hot, one_hot.t())



        # if to_aug:
        outputs_emas = []
        for i in range(self.N):
            _, mena_z = self.model_ema([self.transform(input[0])])
            outputs_ = mena_z.detach()
            outputs_emas.append(outputs_)
        aug_emas = torch.stack(outputs_emas).mean(0)

        raug_emas = aug_emas
        rstandard_ema = standard_ema
        raug_emas[top_k_samples_name] = 0
        rstandard_ema[other_samples_name]=0
        outputs_ema = rstandard_ema + raug_emas


        # Augmentation-averaged Prediction
        # outputs_emas = []
        #
        # # anchor_prob = torch.nn.functional.softmax(anchor_output, dim=1).max(1)[0]
        # # to_aug = anchor_prob.mean(0)<0.1
        #
        #
        # # if to_aug:
        # for i in range(self.N):
        #
        #     _, mena_z = self.model_ema([self.transform(input[0])])
        #     outputs_  = mena_z.detach()
        #     outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        # if to_aug:
        #     outputs_ema = torch.stack(outputs_emas).mean(0)
        # else:
        #     outputs_ema = torch.stack(outputs_emas).mean(0) # standard_ema

        # Augmentation-averaged Prediction

        recon_kl_loss, loss_dict = self.model.get_loss(input, 1, 0.7, 4)
        loss = recon_kl_loss  #0.001*
        #
        # loss += torch.sum(clu_sim * torch.cdist(z, z))  #0.0005 *
        loss += self.consis * (softmax_entropy(z, outputs_ema.detach())).mean(0)
        #z, outputs_ema可以加一个类别信息监督的对比损失，本身以及同类样本为正样本对  其余为负样本
        # loss += self.contra *(self.cl(z, outputs_ema, clu_sim) + self.cl(z, outputs_ema, clu_sim)) / 2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)  #0.999
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.001).float().cuda()

                        with torch.no_grad():
                            # print('p',p)
                            # print('self.model_state[f"{nm}.{npp}"]',self.model_state[f"{nm}.{npp}"])
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        return z, recon_kl_loss, self.consis * (softmax_entropy(z, outputs_ema.detach())).mean(0)  #standard_ema


@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())  #只拷贝模型参数的外层指针，所以修改时原模型参数也会发生改变
    model_anchor = deepcopy(model)  #深度拷贝模型参数，生成新模型，新模型改动时源模型不受影响，但是新模型导数为none
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()  #ema_model中的参数不可求导反传
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():

        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)

    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
