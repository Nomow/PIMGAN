# %%
from utils.stylegan2 import * 
from datasets.UVGan import UVGanDataset1
from metrics.loss import FaceIDLoss
import json
from metrics import metric_main
import time


# %%
device = "cuda:0"
style_mixing_prob = 0
batch_size = 24
num_workers = 4
epochs = 500
data_path = "/src/data/raw_data/ffhq_with_mask"
d_reg_interval = 16
g_reg_interval = 4
ema_kimg = 10
ada_target = 0.6  
ada_interval = 4     
ada_kimg = 300
save_interval_kimgs = 35
stats_collector = training_stats.Collector(regex='.*')
stats_metrics = dict()
run_dir = "/src/current-approach/new/snapshots-stylegan-detected-mask1111122222333333"
os.makedirs(run_dir, exist_ok=True)
stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
stats_interval_kimgs = 1
metric_interval_kimgs = 10
model_save_interval_kimgs = 30


# %%
cfg_dict = {
    "pretrained_model" : "/src/current-approach/new/mse+id_lower/snapshots-stylegan-detected-mask1111122222333333/network-snapshot-001080.pkl",
    "img_size" : 256,
    "cond_dim" : 0,
    "channels" : 3,

    "G_kwargs": {
        "z_dim": 50,
        "w_dim": 512,
        "mapping_kwargs": {
          "num_layers": 8
        },
    "synthesis_kwargs": {
        "channel_base": 16384,
        "channel_max": 512,
        "num_fp16_res": 4,
        "conv_clamp": 256
      }
    }
}

cfg = DefaultMunch.fromDict(cfg_dict)
z_dim = cfg.G_kwargs.z_dim
c_dim = cfg.cond_dim
G, G_ema = create_generator(cfg, device)

# %%
cfg_dict = {
    "pretrained_model" : "/src/current-approach/new/mse+id_lower/snapshots-stylegan-detected-mask1111122222333333/network-snapshot-001080.pkl",
    "img_size" : 256,
    "cond_dim" : 0,
    "channels" : 3,
    "D_kwargs": {
      "block_kwargs": {},
      "mapping_kwargs": {},
      "epilogue_kwargs": {
        "mbstd_group_size": 8
      },
      "channel_base": 16384,
      "channel_max": 512,
      "num_fp16_res": 4,
      "conv_clamp": 256
    },
}

cfg = DefaultMunch.fromDict(cfg_dict)
D = create_discrimnator(cfg, device)

# %%
cfg_dict = {
"p" : 0,
"augment_kwargs": {
    "xflip": 1,
    "rotate90": 1,
    "xint": 1,
    "scale": 1,
    "rotate": 1,
    "aniso": 1,
    "xfrac": 1,
    "brightness": 1,
    "contrast": 1,
    "lumaflip": 1,
    "hue": 1,
    "saturation": 1
  },
}
cfg = DefaultMunch.fromDict(cfg_dict)

augment_pipe = create_augment_pipe(cfg, device)
ada_stats = training_stats.Collector(regex='Loss/signs/real')


# %%

############################
# DECA
############################
cfg_dict = {
  "save_dir" : "./snapshots/snapshot-uv/aligned_snapshot_lpips",
  "device": "cuda:0",
  "random_seed": 0,
  "epochs": 50,
  "DECA_kwargs": {
    "pretrained_model" : '/src/deca/DECA/data/deca_model.tar',
    "topology_path": '/src/deca/DECA/data/head_template.obj',
    'dense_template_path': '/src/deca/DECA/data/texture_data_256.npy',
    'fixed_displacement_path': '/src/deca/DECA/data/fixed_displacement_256.npy',
    'flame_model_path': '/src/deca/DECA/data/generic_model.pkl',
    'flame_lmk_embedding_path': '/src/deca/DECA/data/landmark_embedding.npy',
    'face_mask_path': '/src/deca/DECA/data/uv_face_mask.png',
    'face_eye_mask_path': '/src/deca/DECA/data/uv_face_eye_mask.png',
    'mean_tex_path': '/src/deca/DECA/data/mean_texture.jpg',
    'tex_path': '/src/deca/DECA/data/FLAME_albedo_from_BFM.npz',
    'tex_type': 'BFM',
    'image_size': 224,
    'uv_size': 256,
    'param_list': ['shape', 'tex', 'exp', 'pose', 'cam', 'light'],
    'n_shape': 100,
    'n_tex': 50,
    'n_exp': 50,
    'n_cam': 3,
    'n_pose': 6,
    'n_light': 27,
    'use_tex': True, 
    'jaw_type': 'aa',
    'fr_model_path': '/src/deca/DECA/data/resnet50_ft_weight.pkl', 
    'n_detail': 128, 
    'max_z': 0.01,
    'jaw_type' : 'euler'
  }
}
cfg = DefaultMunch.fromDict(cfg_dict)

flametex = FLAMETex(cfg.DECA_kwargs).to(device)


# %%
cfg_dict = {   
    "snapshot_nimg": 30000,
    "ema_kimg": 10,
    "G_opt_kwargs": {
      "lr": 0.001,
      "betas": [
        0,
        0.99
      ],
      "eps": 1e-08
    },
    "D_opt_kwargs": {
      "lr": 0.0001,
      "betas": [
        0,
        0.99
      ],
      "eps": 1e-08
    },
}

cfg = DefaultMunch.fromDict(cfg_dict)

optimizer_G = torch.optim.Adam(G.parameters(), **cfg.G_opt_kwargs)
optimizer_D = torch.optim.Adam(D.parameters(), **cfg.D_opt_kwargs)


# %%
g_reg_loss = PathLengthLoss(G, device)
r1_reg_loss = R1Regularization(D, device, augment_pipe=augment_pipe)
discrim_loss = DiscriminatorLoss(D, device, augment_pipe_real=augment_pipe)
face_id_loss = FaceIDLoss("r50", "/src/current-approach/synthethic-face-generation-and-manipulation/third_part/backbone.pth", 256, 112).to(device)
mse_loss = torch.nn.MSELoss()


# %%
dataset = UVGanDataset1(data_path)
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)


# %%
batch_idx = 0
cur_nimg = 0
tick_start_nimg = cur_nimg
cur_stats_interval_kimgs = stats_interval_kimgs
cur_model_save_interval_kimgs = model_save_interval_kimgs

for epoch in range(epochs):
    print(epoch)
    for i, data, in enumerate(train_loader):

        #########################################
        # data loading
        real_img = data[0].to(device)
        visibility_mask = data[1].to(device)
        face_eye_mask = data[2].to(device)
        shading_mask = data[3].to(device)
        grid = data[4].to(device)
        gen_lmks = data[5].to(device)
        real_lmks = data[6].to(device)
        z = data[7].to(device)
        c = torch.zeros(batch_size, c_dim).to(device)
        
        #########################################
        # discriminator
        D.zero_grad()

        if((batch_idx+1) % d_reg_interval == 0):
            r1_reg_loss.run(real_img, c)
        else:
            uv_map, ws = run_G(G, z, c, style_mixing_prob)
            rasterized_img = F.grid_sample(uv_map, grid, align_corners=False)
            # gen_img = (face_eye_mask * visibility_mask * rasterized_img ) + (1 - face_eye_mask * visibility_mask) * real_img
            gen_img = (face_eye_mask * visibility_mask * rasterized_img * shading_mask) + (1 - face_eye_mask * visibility_mask) * real_img
            discrim_loss.run(real_img, c, gen_img, c)
        optimizer_D.step()

        
        #########################################
        # generator
        G.zero_grad()

        if((batch_idx+ 1) % g_reg_interval == 0):
            g_reg_loss.run(z, c)
        else:
            uv_map, ws = run_G(G, z, c, style_mixing_prob)
            rasterized_img = F.grid_sample(uv_map, grid, align_corners=False)
            gen_img = (face_eye_mask * visibility_mask * rasterized_img * shading_mask) + (1 - face_eye_mask * visibility_mask) * real_img
            # gen_img = (face_eye_mask * visibility_mask * rasterized_img ) + (1 - face_eye_mask * visibility_mask) * real_img

            # gen loss
            gen_logits = run_D(D, gen_img, c, None)
            gen_loss = torch.nn.functional.softplus(-gen_logits).mean()
             
            # id loss
            id_loss = 10 * face_id_loss(torch.clip(gen_img, min=0, max=1), real_img, gen_lmks, real_lmks)
            
            
            b = flametex.texture_basis.squeeze(0)
            bt = flametex.texture_basis.squeeze(0).T
            a_mean = flametex.texture_mean.squeeze(0)
            precision_matrix = torch.inverse(torch.matmul(bt, b)) 
            resized_uv_map = F.interpolate(uv_map, [512, 512]).reshape(len(uv_map), -1)
            albedo_loss = torch.linalg.norm(torch.matmul(torch.matmul(precision_matrix, bt), (a_mean - resized_uv_map).T), axis=0).mean()
            # l2 loss
            l2_loss = 1000 *  mse_loss(real_img, gen_img)
            total_loss = gen_loss + id_loss + l2_loss + albedo_loss
            total_loss.backward()
            training_stats.report('/Loss/G/loss', total_loss)
            training_stats.report('/Loss/G/l2_loss', l2_loss)
            training_stats.report('/Loss/G/id_loss', id_loss)
            training_stats.report('/Loss/G/gen_loss', gen_loss)
            training_stats.report('/Loss/G/albedo_loss', albedo_loss)

        optimizer_G.step()

        
        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            G_ema.requires_grad_(False)
            G.requires_grad_(False)
            ema_nimg = ema_kimg * 1000
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G.requires_grad_(True)


        # updates augmentation probability
        if ((batch_idx + 1) % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # statistics update
        if(cur_nimg > cur_stats_interval_kimgs * 1000):
            cur_stats_interval_kimgs = cur_stats_interval_kimgs + stats_interval_kimgs
            stats_collector.update()
            stats_dict = stats_collector.as_dict()
            timestamp = time.time()
            if stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                stats_jsonl.write(json.dumps(fields) + '\n')
                stats_jsonl.flush()
            torchvision.utils.save_image(torchvision.utils.make_grid(gen_img.detach().cpu()), os.path.join(run_dir, f'fakes.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(real_img.detach().cpu()), os.path.join(run_dir, f'real.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(uv_map.detach().cpu()), os.path.join(run_dir, f'uv.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(rasterized_img.detach().cpu()), os.path.join(run_dir, f'rastered.png'))

        if(cur_nimg > cur_model_save_interval_kimgs * 1000):
            cur_model_save_interval_kimgs = cur_model_save_interval_kimgs + model_save_interval_kimgs
            # Save network snapshot.
            snapshot_data = {}
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
            del snapshot_data
            torchvision.utils.save_image(torchvision.utils.make_grid(gen_img.detach().cpu()), os.path.join(run_dir, f'{cur_nimg//1000:06d}fakes.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(real_img.detach().cpu()), os.path.join(run_dir, f'{cur_nimg//1000:06d}real.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(uv_map.detach().cpu()), os.path.join(run_dir, f'{cur_nimg//1000:06d}uv.png'))
            torchvision.utils.save_image(torchvision.utils.make_grid(rasterized_img.detach().cpu()), os.path.join(run_dir, f'{cur_nimg//1000:06d}rastered.png'))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1


        







        



