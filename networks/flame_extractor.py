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


class Extractor(nn.Module):
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

        self._create_model(self.cfg)
        self._setup_renderer(self.cfg)

    def _setup_renderer(self, cfg):
        self.render = SRenderY(448, obj_filename=cfg.DECA_kwargs.topology_path, uv_size=cfg.DECA_kwargs.uv_size, rasterizer_type="pytorch3d").to(self.device)
        # face mask for rendering details
        mask = imread(cfg.DECA_kwargs.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [cfg.DECA_kwargs.uv_size, cfg.DECA_kwargs.uv_size]).to(self.device)
        mask = imread(cfg.DECA_kwargs.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [cfg.DECA_kwargs.uv_size, cfg.DECA_kwargs.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(cfg.DECA_kwargs.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(cfg.DECA_kwargs.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [cfg.DECA_kwargs.uv_size, cfg.DECA_kwargs.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(cfg.DECA_kwargs.dense_template_path, allow_pickle=True, encoding='latin1').item()


    def _create_model(self, cfg):
        # DECA set up parameters
        self.n_param = cfg.DECA_kwargs.n_shape+cfg.DECA_kwargs.n_tex+cfg.DECA_kwargs.n_exp+cfg.DECA_kwargs.n_pose+cfg.DECA_kwargs.n_cam+cfg.DECA_kwargs.n_light
        self.n_detail = cfg.DECA_kwargs.n_detail
        self.n_cond = cfg.DECA_kwargs.n_exp + 3 # exp + jaw pose
        self.num_list = [cfg.DECA_kwargs.n_shape, cfg.DECA_kwargs.n_tex, cfg.DECA_kwargs.n_exp, cfg.DECA_kwargs.n_pose, cfg.DECA_kwargs.n_cam, cfg.DECA_kwargs.n_light]
        self.param_dict = {i:cfg.DECA_kwargs.get('n_' + i) for i in cfg.DECA_kwargs.param_list}

        # DECA encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)

        # DECA decoders
        self.flame = FLAME(cfg.DECA_kwargs).to(self.device)
        if cfg.DECA_kwargs.use_tex:
            self.flametex = FLAMETex(cfg.DECA_kwargs).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, out_scale=cfg.DECA_kwargs.max_z, sample_mode = 'bilinear').to(self.device)
        

        # resume deca model
        model_path_deca = self.cfg.DECA_kwargs.pretrained_model
        if os.path.exists(model_path_deca):
            print(f'DECA trained model found. load {model_path_deca}')
            checkpoint = torch.load(model_path_deca)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'DECA trained model not found please check model path: {model_path_deca}')

        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

        

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + uv_coarse_normals*(1-self.uv_face_eye_mask)
        return uv_detail_normals

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    def encode(self, images):
        parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        detailcode = self.E_detail(images)
        codedict['detail'] = detailcode
        if self.cfg.DECA_kwargs.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose  
        return codedict
    def render_orig(self, vertices, transformed_vertices, albedos=None, images=None, detail_normal_images=None, 
                lights=None, h=None, w=None, light_type = 'point'):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1)); face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1)); transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), 
                        transformed_face_normals.detach(), 
                        face_vertices.detach(), 
                        face_normals,
                        # self.face_uvcoords.expand(batch_size, -1, -1, -1)
                        ], 
                        -1)
        # rasterize
        # import ipdb; ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]; grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        outputs = {
            'images': images*alpha_images,
            'albedo_images': albedo_images*alpha_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }
        
        return outputs

