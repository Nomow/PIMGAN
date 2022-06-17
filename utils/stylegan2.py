# %%
import copy
import dnnlib
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from networks import legacy
from networks.stylegan2 import Generator as Stylegan2Generator
from networks.stylegan2 import Discriminator as Stylegan2Discriminator
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from networks.encoders import ResnetEncoder
from networks.FLAME import FLAME, FLAMETex
from networks.decoders import Generator
from utils import util
from utils.rotation_converter import batch_euler2axis
from utils.config import cfg
torch.backends.cudnn.benchmark = True
from utils.renderer import SRenderY
import torch
import torch.nn as nn
# %%
import math
import random
import os
import copy
import pickle
import numpy as np
import torch

import os, sys
import torch
import torchvision
import torch.nn.functional as F

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from utils.rotation_converter import batch_euler2axis
from utils.tensor_cropper import transform_points
from utils.config import cfg
torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from networks.UVGan import UVGan
from munch import DefaultMunch
import warnings
warnings.filterwarnings('ignore')

import albumentations as albu
import albumentations.pytorch
import logging
from torch.optim.lr_scheduler import MultiStepLR
from networks.augment import AugmentPipe

torch.backends.cudnn.benchmark = True    # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
torch.backends.cudnn.allow_tf32 = False        # Allow PyTorch to internally use tf32 for convolutions
conv2d_gradfix.enabled = True                       # Improves training speed.
grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.



# %%
import copy
import dnnlib
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from networks import legacy
from networks.stylegan2 import Generator as Stylegan2Generator
from networks.stylegan2 import Discriminator as Stylegan2Discriminator
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from networks.encoders import ResnetEncoder
from networks.FLAME import FLAME, FLAMETex
from networks.decoders import Generator
from utils import util
from utils.rotation_converter import batch_euler2axis
from utils.config import cfg
torch.backends.cudnn.benchmark = True
from utils.renderer import SRenderY
import torch
import torch.nn as nn
import torch.autograd

def create_generator(cfg, device, train=True):
    common_kwargs_G = dict(c_dim=cfg.cond_dim, img_resolution=cfg.img_size, img_channels=cfg.channels)
    G = Stylegan2Generator(**cfg.G_kwargs, **common_kwargs_G).requires_grad_(False).to(device) 
    G_ema = copy.deepcopy(G).to(device) 
    if os.path.exists(cfg.pretrained_model):
        print(f'Stylegan generator trained model found. load {cfg.pretrained_model}')
        with dnnlib.util.open_url(cfg.pretrained_model) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', None), ('G_ema', G_ema)]:
            if(module is not None):
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    else:
        print(f'Stylegan generator trained model not found found, creating new model {cfg.pretrained_model}')
    
    if(train == True):
        G.train().requires_grad_(True)
        G_ema.train().requires_grad_(True)
    else:
        G.eval().requires_grad_(False)
        G_ema.eval().requires_grad_(False)
    return G, G_ema

    
def create_discrimnator(cfg, device, train=True):
    common_kwargs_D = dict(c_dim=cfg.cond_dim, img_resolution=cfg.img_size, img_channels=cfg.channels)
    D = Stylegan2Discriminator(**cfg.D_kwargs, **common_kwargs_D).requires_grad_(False).to(device) 

    if os.path.exists(cfg.pretrained_model):
        print(f'Stylegan discrimnator trained model found. load {cfg.pretrained_model}')
        with dnnlib.util.open_url(cfg.pretrained_model) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', None), ('D', D), ('G_ema', None)]:
          if(module is not None):
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    else:
        print(f'Stylegan discriminator trained model not found found, creating new model {cfg.pretrained_model}')
    
    if(train == True):
        D.train().requires_grad_(True)
    else:
        D.eval().requires_grad_(False)
    return D

def create_augment_pipe(cfg, device):
    augment = AugmentPipe(**cfg.augment_kwargs).requires_grad_(False).to(device)
    augment.p.copy_(torch.as_tensor(0))
    return augment

def run_G(G, z, c, style_mixing_prob):
    ws = G.mapping(z, c)
    if style_mixing_prob > 0:
        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
        cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
        ws[:, cutoff:] = G.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
    img = G.synthesis(ws)
    return img, ws


from torch_utils import training_stats

class Loss:
    def run(self, gen_z, gen_c): # to be overridden by subclass
        raise NotImplementedError()


class PathLengthLoss(Loss):
    def __init__(self, G, device, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, name = ""):
        super().__init__()
        self.G = G
        self.device = device
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.name = name
        
    def run(self, gen_z, gen_c):
        with torch.autograd.profiler.record_function('Gpl_forward'):
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            gen_img, gen_ws = run_G(self.G, gen_z[:batch_size], gen_c[:batch_size], self.style_mixing_prob)
            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            training_stats.report(self.name + '/Loss/pl_penalty', pl_penalty)
            loss_Gpl = pl_penalty * self.pl_weight
            training_stats.report(self.name + '/Loss/G/reg', loss_Gpl)
        with torch.autograd.profiler.record_function('Gpl_backward'):
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().backward()

def run_D(D, img, c, augment_pipe):
    if augment_pipe is not None:
        img = augment_pipe(img)
    logits = D(img, c)
    return logits

class R1Regularization(Loss):
    def __init__(self, D, device, augment_pipe=None, r1_gamma=10, name = ""):
        super().__init__()
        self.device = device
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.pl_mean = torch.zeros([], device=device)
        self.name = name

    def run(self, real_img, real_c):

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        name = 'Dr1'
        with torch.autograd.profiler.record_function(name + '_forward'):
            real_img_tmp = real_img.detach().requires_grad_(True)
            real_logits = run_D(self.D, real_img_tmp, real_c, self.augment_pipe)

            loss_Dr1 = 0
            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1,2,3])
            loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
            training_stats.report(self.name + '/Loss/r1_penalty', r1_penalty)
            training_stats.report(self.name + '/Loss/D/reg', loss_Dr1)

        with torch.autograd.profiler.record_function(name + '_backward'):
            (real_logits * 0  + loss_Dr1).mean().backward()


class DiscriminatorLoss(Loss):
    def __init__(self, D, device, augment_pipe_real=None, augment_pipe_fake=None, name = ""):
        super().__init__()
        self.device = device
        self.D = D
        self.augment_pipe_real = augment_pipe_real
        self.augment_pipe_fake = augment_pipe_fake
        self.name = name

    def run(self, real_img, real_c, gen_img, gen_c):

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        with torch.autograd.profiler.record_function('Dgen_forward'):
            gen_logits = run_D(self.D, gen_img, gen_c, augment_pipe=self.augment_pipe_fake) # Gets synced by loss_Dreal.
            training_stats.report(self.name +'/Loss/scores/fake', gen_logits)
            training_stats.report(self.name +'/Loss/signs/fake', gen_logits.sign())
            loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
        with torch.autograd.profiler.record_function('Dgen_backward'):
            loss_Dgen.mean().backward()

        # Dmain: Maximize logits for real images.
        name = 'Dreal'
        with torch.autograd.profiler.record_function(name + '_forward'):
            real_img_tmp = real_img.detach().requires_grad_(False)
            real_logits = run_D(self.D, real_img_tmp, real_c, augment_pipe=self.augment_pipe_real)
            training_stats.report(self.name + 'Loss/scores/real', real_logits)
            training_stats.report(self.name + 'Loss/signs/real', real_logits.sign())

            loss_Dreal = 0
            loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
            training_stats.report(self.name + 'Loss/D/loss', loss_Dgen + loss_Dreal)

        with torch.autograd.profiler.record_function(name + '_backward'):
            loss_Dreal.mean().backward()

class GeneratorLoss(Loss):
    def __init__(self, D, device, augment_pipe_fake=None, name = ""):
        super().__init__()
        self.device = device
        self.D = D
        self.augment_pipe_fake = augment_pipe_fake
        self.name = name

    def run(self, gen_img, gen_c):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            gen_logits = run_D(self.D, gen_img, gen_c, augment_pipe=self.augment_pipe_fake)
            loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
            training_stats.report(self.name + '/Loss/G/loss', loss_Gmain)
        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Gmain.mean().backward()

