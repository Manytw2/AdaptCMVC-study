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


def get_tta_transforms(gaussian_std: float=0.005,CROP_SIZE=64, soft=True, clip_inputs=False):
    img_shape = (CROP_SIZE, CROP_SIZE, 1) #coil20

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
    def __init__(self, model, optimizer, steps=1, episodic=False,num_classes=20,CROP_SIZE=64,contra=1.0,consis=1.0,N=2,sample_num_each_clusters=10):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms(0.005,CROP_SIZE, False, False)
        self.num_classes = num_classes

        self.contra = contra
        self.consis = consis
        self.N = N
        self.sample_num_each_clusters = sample_num_each_clusters
        self.similarity = nn.CosineSimilarity(dim=2)


    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs,rec_loss,consis_loss,stru_loss = self.forward_and_adapt(x, self.optimizer)

        return outputs,rec_loss,consis_loss,stru_loss

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
        simmatrix_path = ""
        input = x[0]
        last_pre = x[1]
        source_center = x[2]
        clu_sim = x[3]
        recon, z = self.model(input)
        _, standard_ema = self.model_ema(input)
        self.model_ema.train()
        # Teacher Prediction
        _, anchor_output = self.model_anchor(input)
        distances_to_cluster_centers = np.linalg.norm(z.cpu().detach().numpy() - source_center[last_pre.cpu().detach().numpy()], axis=1)

        outputs_emas = []
        for i in range(self.N):
            _, mena_z = self.model_ema([self.transform(input[0])])#[top_k_samples_name]
            outputs_ = mena_z.detach()
            outputs_emas.append(outputs_)
        aug_emas = torch.stack(outputs_emas).mean(0)


        outputs_ema = aug_emas




        recon_kl_loss, loss_dict = self.model.get_loss(input, 1, 0.7, 4)
        loss = recon_kl_loss

        sim_weight = 1 / torch.from_numpy(distances_to_cluster_centers).cuda()**2
        weight_consis_loss = self.consis * (sim_weight * softmax_entropy(z, outputs_ema.detach())).mean(0)

        loss += weight_consis_loss

        cos_sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0))
        stru_loss = self.contra * F.mse_loss(clu_sim, cos_sim, reduction='none').mean(1).mean(0)
        loss += stru_loss


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Teacher update
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=0.999)  #0.999



        return z, recon_kl_loss, weight_consis_loss, stru_loss


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
