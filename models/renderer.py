import torch
import numpy as np
import mcubes


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class NeuSRenderer:
    def __init__(
        self,
        neus_model,
        n_samples,
        n_importance,
        perturb,
        **kwargs,
    ):
        self.neus_model = neus_model
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb

    def render(self, rays_o, rays_d, near, far):
        ret_fine = self.neus_model.forward_(rays_o, rays_d, near, far)

        return {
            'color_fine': ret_fine['color'],
            's_val': ret_fine['s_val'],
            'opacity': ret_fine['opacity'],
            'gradients': ret_fine['gradients'],
            'num_samples': ret_fine['num_samples'],
            'rays_valid': ret_fine['rays_valid'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.neus_model.geometry.sdf(pts)
        )
