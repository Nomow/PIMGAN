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
# from utils.renderer import SRenderY
import torch
import torch.nn as nn


class UVGan(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super().__init__()

        self.cfg = config
        self.device = device
        self.image_size = self.cfg.DECA_kwargs.image_size
        self.image_channels = 3
        self.condition_dim = 0
        self.uv_size = self.cfg.DECA_kwargs.uv_size

        mask = cv2.imread(self.cfg.DECA_kwargs.face_eye_mask_path).astype(np.float32)/255.
        mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [self.uv_size, self.uv_size]).to(self.device)

        torch.backends.cudnn.benchmark = True    # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = False        # Allow PyTorch to internally use tf32 for convolutions
        conv2d_gradfix.enabled = True                       # Improves training speed.
        grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

        self._create_model(self.cfg)



    def _create_model(self, cfg):
        # DECA set up parameters
        self.n_param = cfg.DECA_kwargs.n_shape+cfg.DECA_kwargs.n_tex+cfg.DECA_kwargs.n_exp+cfg.DECA_kwargs.n_pose+cfg.DECA_kwargs.n_cam+cfg.DECA_kwargs.n_light
        self.n_detail = cfg.DECA_kwargs.n_detail
        self.n_cond = cfg.DECA_kwargs.n_exp + 3 # exp + jaw pose
        self.num_list = [cfg.DECA_kwargs.n_shape, cfg.DECA_kwargs.n_tex, cfg.DECA_kwargs.n_exp, cfg.DECA_kwargs.n_pose, cfg.DECA_kwargs.n_cam, cfg.DECA_kwargs.n_light]


        self.flametex = FLAMETex(cfg.DECA_kwargs).to(self.device)
        


        #Stylegan2
        common_kwargs_G = dict(c_dim=self.condition_dim, img_resolution=self.uv_size, img_channels=self.image_channels)
        self.generator_stylegan = Stylegan2Generator(**cfg.stylegan2.G_kwargs, **common_kwargs_G).requires_grad_(False).to(self.device) 
       
        common_kwargs_D = dict(c_dim=self.condition_dim, img_resolution=self.uv_size, img_channels=self.image_channels)
        self.discriminator_stylegan = Stylegan2Discriminator(**cfg.stylegan2.D_kwargs, **common_kwargs_D).requires_grad_(False).to(self.device) 
        self.generator_stylegan_ema = copy.deepcopy(self.generator_stylegan).to(self.device) 

        # resume stylegan2 model
        model_path_stylegan = self.cfg.stylegan2.pretrained_model
        if os.path.exists(model_path_stylegan):
            print(f'Stylegan trained model found. load {model_path_stylegan}')
            with dnnlib.util.open_url(model_path_stylegan) as f:
                resume_data = legacy.load_network_pkl(f)
            for name, module in [('G', self.generator_stylegan), ('D', self.discriminator_stylegan), ('G_ema', self.generator_stylegan_ema)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        else:
            print(f'Stylegan trained model not found found please check model path: {model_path_stylegan}')

        # eval mode
        self.generator_stylegan.eval()
        self.discriminator_stylegan.eval()
        self.generator_stylegan_ema.eval()
        

    # def decompose_code(self, code, num_dict):
    #     ''' Convert a flattened parameter vector to a dictionary of parameters
    #     code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    #     '''
    #     code_dict = {}
    #     start = 0
    #     for key in num_dict:
    #         end = start+int(num_dict[key])
    #         code_dict[key] = code[:, start:end]
    #         start = end
    #         if key == 'light':
    #             code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
    #     return code_dict

    # def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
    #     ''' Convert displacement map into detail normal map
    #     '''
    #     batch_size = uv_z.shape[0]
    #     uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
    #     uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
    #     uv_z = uv_z*self.uv_face_eye_mask
    #     uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
    #     dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
    #     uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
    #     uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
    #     uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
    #     return uv_detail_normals

    # def visofp(self, normals):
    #     ''' visibility of keypoints, based on the normal direction
    #     '''
    #     normals68 = self.flame.seletec_3d68(normals)
    #     vis68 = (normals68[:,:,2:] < 0.1).float()
    #     return vis68

    # def encode(self, images):
    #     parameters = self.E_flame(images)
    #     codedict = self.decompose_code(parameters, self.param_dict)
    #     codedict['images'] = images
    #     detailcode = self.E_detail(images)
    #     codedict['detail'] = detailcode
    #     if self.cfg.DECA_kwargs.jaw_type == 'euler':
    #         posecode = codedict['pose']
    #         euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
    #         posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
    #         codedict['pose'] = posecode
    #         codedict['euler_jaw_pose'] = euler_jaw_pose  
    #     return codedict
