import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import scale_anything
from .fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from nerfacc import ContractionType, OccupancyGrid, ray_marching, rendering


class NeuSModel(nn.Module,):
    def __init__(self, config, scene_aabb, device):
        super().__init__()
        self.conf = config
        self.training = config['general']['training']
        self.device = device
        self.grid_prune = config['train.grid_prune']
        self.occ_thre = config['train'].get('occ_thre', 0.01)
        self.randomized = config['train.randomized']
        self.num_samples_per_ray = config['train.num_samples_per_ray']
        self.resolution_occ_grid = config.get('train.resolution_occ_grid', 128)
        self.cos_anneal_ratio = 0.0

        self.geometry = SDFNetwork(
            **self.conf['model.sdf_network'],
            device=self.device,
        ).to(self.device)
        self.texture = RenderingNetwork(
            **self.conf['model.rendering_network'],
            device=self.device,
        ).to(self.device)
        self.variance = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        self.register_buffer('scene_aabb', scene_aabb.float())
        if self.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=self.resolution_occ_grid,
                contraction_type=ContractionType.AABB
            ).to(self.device)
        self.register_buffer('background_color', torch.as_tensor([0, 0, 0], dtype=torch.float32), persistent=False)
        self.radius = (abs(scene_aabb.max()) + abs(scene_aabb.min()) ) / 2.0
        self.render_step_size = 1.732 * 2 * self.radius / (self.num_samples_per_ray)

    def get_params_to_train(self, lr_list):
        params_to_train = [
            {'params': params, 'lr': lr_list[i]} for i, params in enumerate([
                    self.variance.parameters(), self.geometry.parameters(), self.texture.parameters(),
            ])
        ]
        return params_to_train

    def update_step(self, epoch: int, global_step: int):
        cos_anneal_end = self.conf.get('train.anneal_end', 0,)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            x = scale_anything(x, self.scene_aabb, (0, 1))
            sdf = self.geometry.sdf(x, training=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf - self.render_step_size * 0.5
            estimated_prev_sdf = sdf + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha

        if self.training and self.grid_prune:
            self.occupancy_grid.every_n_step(
                step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.occ_thre
            )

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_(self, rays_o, rays_d, near, far):
        sdf_grad_samples = []

        def alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            positions = scale_anything(positions, self.scene_aabb, (0, 1))
            sdf = self.geometry.sdf(positions,)
            sdf_grad = self.geometry.gradient(positions,).squeeze()
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            dists = t_ends - t_starts
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            return alpha[...,None]

        def rgb_alpha_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            midpoints = (t_starts + t_ends) / 2.
            positions = t_origins + t_dirs * midpoints
            positions = scale_anything(positions, self.scene_aabb, (0, 1))
            geometry = self.geometry(positions,)
            sdf, feature = geometry[:, :1], geometry[:, 1:]
            sdf_grad = self.geometry.gradient(positions,)
            sdf_grad_samples.append(sdf_grad)
            dists = t_ends - t_starts
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            alpha = self.get_alpha(sdf, normal, t_dirs, dists)
            rgb = self.texture(positions, normal, t_dirs, feature)
            return rgb, alpha[..., None]

        with torch.no_grad():
            packed_info, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.grid_prune else None,
                alpha_fn=alpha_fn,
                near_plane=near.squeeze(), far_plane=far.squeeze(),
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0,
            )

        rgb, opacity, depth = rendering(
            ray_indices=packed_info,
            t_starts=t_starts,
            t_ends=t_ends,
            n_rays=len(rays_o),
            rgb_alpha_fn=rgb_alpha_fn,
        )

        sdf_grad_samples = torch.cat(sdf_grad_samples, dim=0)
        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)
        num_samples = torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device)

        rv = {
            'color': rgb,
            'opacity': opacity,
            'rays_valid': opacity > 0,
            'gradients': sdf_grad_samples,
            'num_samples': num_samples,
            's_val': 1 / self.variance.inv_s,
        }

        if self.training:
            rv.update({
                'gradients': sdf_grad_samples,
            })

        return rv
