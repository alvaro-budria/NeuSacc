import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from pynvml import *

import wandb

import torch
import torch.nn.functional as F

from models.dataset import Dataset
from models.neus import NeuSModel
from models.renderer import NeuSRenderer
import ops.mesh as mesh_ops
from ops.mesh import metrics
from ops.loss import binary_cross_entropy


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, args=None):
        self.device = torch.device(args.gpu)

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.use_profiler = self.conf['general.use_profiler']
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.train_num_rays = self.conf.get_int('train.train_num_rays')
        self.train_num_samples = self.train_num_rays * self.conf.get('train.num_samples_per_ray')
        self.max_train_num_rays = self.conf.get_int('train.max_train_num_rays')
        self.dynamic_ray_sampling = self.conf.get_bool('train.dynamic_ray_sampling')
        self.iters_to_accumulate = self.conf.get_int('train.iters_to_accumulate', 1)
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_variance = self.conf.get_float('train.learning_rate_variance', self.learning_rate)
        self.learning_rate_geometry = self.conf.get_float('train.learning_rate_geometry', self.learning_rate)
        self.learning_rate_texture = self.conf.get_float('train.learning_rate_texture', self.learning_rate)
        self.scheduler_type = self.conf.get_string('train.scheduler_type', default="cosine")
        self.decay_rate = self.conf.get_float('train.decay_rate', default=0.98)
        self.learning_rate_low = self.conf.get_float('train.learning_rate_low', default=1e-5)
        self.beta1 = self.conf.get_float('train.beta1')
        self.beta2 = self.conf.get_float('train.beta2')
        self.eps = self.conf.get_float('train.eps')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.color_weight = self.conf.get_float('train.color_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Scene
        self.scene_aabb = torch.cat(
            [
                torch.from_numpy(self.dataset.object_bbox_min),
                torch.from_numpy(self.dataset.object_bbox_max),
            ]
        ).float().to(self.device)

        # Networks
        self.neus_model = NeuSModel(self.conf, self.scene_aabb, self.device)

        # Optimizer
        self.lr_list = [
            self.learning_rate_variance,
            self.learning_rate_geometry,
            self.learning_rate_texture,
        ]
        params_to_train = self.neus_model.get_params_to_train(lr_list=self.lr_list,)
        self.optimizer = torch.optim.Adam(
            params_to_train,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.conf.get_float('train.weight_decay', default=1e-7),
        )
        self.grad_scaler = torch.cuda.amp.GradScaler(
            init_scale=2**11, growth_factor=1.5, backoff_factor=0.5, growth_interval=8000, enabled=True,
        )
        self.renderer = NeuSRenderer(
            self.neus_model,
            **self.conf['model.neus_renderer'],
        )

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

        # Logging
        os.environ["WANDB_DIR"] = "../"
        self.run = wandb.init(
            project="NeuS-Enhanced", save_code=True, dir=self.base_exp_dir,
        )
        wandb.watch(
            (self.neus_model),
            log="all",
            log_freq=100,
            criterion=None,
        )

    def train(self):
        self.start_profiler()
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        with self.run:
            for iter_i in tqdm(range(res_step)):
                self.optimizer.zero_grad()

                self.neus_model.update_step(-1, iter_i)

                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.train_num_rays)

                rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).float()
                else:
                    mask = torch.ones_like(mask)

                mask_sum = mask.sum() + 1e-5

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    render_out = self.renderer.render(rays_o, rays_d, near, far)

                    color_fine = render_out['color_fine']
                    s_val = render_out['s_val']
                    gradients = render_out['gradients']
                    opacity = torch.clamp(render_out['opacity'], 1.e-3, 1.-1.e-3)

                    # Loss
                    color_fine_loss = F.smooth_l1_loss(
                        color_fine[render_out['rays_valid']], true_rgb[render_out['rays_valid']], reduction='sum'
                    ) / mask_sum
                    psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                    norm_unscaled_gradients = torch.linalg.norm(gradients, ord=2, dim=-1,)
                    eikonal_loss = ((norm_unscaled_gradients - 1.)**2).mean()

                    mask_loss = binary_cross_entropy(opacity, mask.squeeze())

                    loss = self.color_weight * color_fine_loss + eikonal_loss * self.igr_weight + mask_loss * self.mask_weight

                self.grad_scaler.scale(loss).backward()

                if (iter_i + 1) % self.iters_to_accumulate == 0:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.iter_step += 1
                    self.run.log(
                        {
                            "Loss/loss": loss,
                            "Loss/color_loss": color_fine_loss,
                            "Loss/eikonal_loss": eikonal_loss,
                            "Statistics/s_val": s_val,
                            "Statistics/psnr": psnr,
                            "epoch": self.iter_step,
                            "Statistics/lr0": self.optimizer.param_groups[0]['lr'],
                            "Statistics/lr1": self.optimizer.param_groups[1]['lr'],
                            "Statistics/lr2": self.optimizer.param_groups[2]['lr'],
                        }
                    )

                if self.dynamic_ray_sampling:
                    train_num_rays = int(
                        self.train_num_rays * (
                            self.train_num_samples / (1+render_out['num_samples'].sum().item())
                        )
                    )
                    self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.max_train_num_rays)

                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir, flush=True)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']), flush=True)
                    print([self.optimizer.param_groups[i]['lr'] for i in range(3)])

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image()

                if self.iter_step % self.val_mesh_freq == 0:
                    try:
                        self.validate_mesh()
                    except Exception as e:
                        print("Validation mesh failed: ", e)

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

                self.step_profiler()
                if self.use_profiler and self.iter_step >= self.profiler_end_iter:
                    break

            self.stop_profiler()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images).cpu()

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
            for i, g in enumerate(self.optimizer.param_groups):
                g['lr'] = self.lr_list[i] * learning_factor
        else:
            if self.scheduler_type == "cosine":
                alpha = self.learning_rate_alpha
                progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
                for g in self.optimizer.param_groups:
                    if g['lr'] >= self.learning_rate_low:
                        g['lr'] = self.lr_list[i] * learning_factor
            elif self.scheduler_type == "ExponentialLR":
                for g in self.optimizer.param_groups:
                    if g['lr'] >= self.learning_rate_low:
                        g['lr'] = g['lr'] * self.decay_rate

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'NeuSModel': self.neus_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx), flush=True)

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.train_num_rays)
        rays_d = rays_d.reshape(-1, 3).split(self.train_num_rays)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.train_num_rays)
        rays_d = rays_d.reshape(-1, 3).split(self.train_num_rays)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            render_out = self.renderer.render(rays_o_batch, rays_d_batch,)
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=512, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        if len(triangles) > 0:
            # Compute L2 chamfer distance
            gt_mesh_path = os.path.join(self.dataset.data_dir, "mesh_gt.obj")
            print(f"Fetching mesh in {gt_mesh_path}", flush=True)

            chamfer_gt_pred, chamfer_pred_gt, _, mesh_pred_aligned = metrics.measure_chamfer([vertices, triangles], gt_mesh_path, p=2)
            chamfer_gt = np.mean(chamfer_gt_pred)
            self.run.log(
                {
                    "Statistics/L2_chamfer": chamfer_gt,
                    "epoch": self.iter_step,
                }
            )

            print('iter:{:8>d} L2_chamfer = {}'.format(self.iter_step, chamfer_gt), flush=True)

            C_chamfer = viz.error_to_color(chamfer_pred_gt, clipping_error=True)
            mesh_ops.save(
                os.path.join(self.base_exp_dir, f'{self.iter_step:06d}_chamfer_pred.obj'),
                V=mesh_pred_aligned[0],
                F=mesh_pred_aligned[1],
                C=C_chamfer,
            )

            # compute V2S distance
            mean_v2s, v2s = metrics.measure_v2s([vertices, triangles], gt_mesh_path, device=self.device)
            self.run.log(
                {
                    "Statistics/V2S": mean_v2s,
                    "epoch": self.iter_step,
                }
            )
            print('iter:{:8>d} V2S = {}'.format(self.iter_step, mean_v2s), flush=True)

            C_v2s = viz.error_to_color(v2s, clipping_error=True)
            mesh_ops.save(
                os.path.join(self.base_exp_dir, f'{self.iter_step:06d}_V2S_pred.obj'),
                V=mesh_pred_aligned[0],
                F=mesh_pred_aligned[1],
                C=C_v2s,
            )

            logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def start_profiler(self):
        if self.use_profiler:
            wait, warmup, active, repeat = 4, 4, 1, 1
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=os.path.join(self.base_exp_dir, 'logs')),
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            self.profiler_end_iter = (wait + warmup + active) * repeat
            self.prof.start()

    def stop_profiler(self):
        if self.use_profiler:
            self.prof.stop()

    def step_profiler(self):
        if self.use_profiler:
            self.prof.step()


if __name__ == '__main__':
    print('Hello Wooden', flush=True)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Device: {torch.cuda.get_device_name(device=args.gpu)}")
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f"total GPU MEM   : {info.total}", flush=True)
        print(f"free  GPU MEM   : {info.free}", flush=True)
        print(f"used  GPU MEM   : {info.used}", flush=True)

    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
