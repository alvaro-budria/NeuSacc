import torch
import torch.nn as nn
import numpy as np
from models.model_components import get_mlp, get_encoding


class SDFNetwork(nn.Module):

    def __init__(
        self,
        geo_feat_dim      : int = 128,
        d_hidden          : int = 128,
        n_layers          : int = 1,
        n_frequencies     : int = 6,
        include_xyz       : bool = True,
        sphere_init       : bool = True,
        sphere_init_radius: float = 0.5,
        weight_norm       : bool = True,

        num_lods          : int   = 16,
        feature_dim       : int   = 2,
        codebook_bitwidth : int   = 18,
        max_width_codebook: int = 2**12,
        min_width_codebook: int = 16,

        device            : str   = 'cuda',
        **kwargs,
    ):
        super().__init__()

        self.__hidden__ = torch.nn.Linear(3, 1, bias=False)
        self.device = device
        per_level_scale = np.exp((np.log(max_width_codebook) - np.log(min_width_codebook)) / num_lods)

        if num_lods > 0:
            encoding_config={
                "otype": "HashGrid",
                "hash": "CoherentPrime",
                "n_levels": num_lods,
                "n_features_per_level": feature_dim,
                "log2_hashmap_size": codebook_bitwidth,
                "base_resolution": min_width_codebook,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
                "include_xyz": include_xyz,
                "n_frequencies": n_frequencies,
            }
            self.encoding_sdf = get_encoding(3, encoding_config, device=self.device)
        else :
            self.encoding_sdf = get_encoding(
                3,
                {"otype": "VanillaFrequency", "n_frequencies": n_frequencies,},
                device=self.device
            )
        self.mlp_sdf = get_mlp(
            n_input_dims=self.encoding_sdf.n_output_dims,
            n_output_dims=1 + geo_feat_dim,
            config={
                "otype": "VanillaMLP",
                "n_neurons": d_hidden,
                "n_hidden_layers": n_layers,
                "sphere_init": sphere_init,
                "sphere_init_radius": sphere_init_radius,
                "weight_norm": weight_norm,
                "output_activation": "Identity",
            },
            device=self.device
        )

        self.mlp_sdf = torch.nn.Sequential(*[self.encoding_sdf, self.mlp_sdf])
        self.print_n_params("SDFNetwork")
        torch.cuda.empty_cache()

    def print_n_params(self, name):
        n_params = sum(p.numel() for p in self.parameters())
        n_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"* {name} * - #Params = {n_params}; #Trainable Params = {n_train_params}", flush=True)

    def forward(self, inputs, training=True):
        if not training:
            inputs = inputs.clone() # inputs may be in inference mode, get a copy to enable grad
        inputs = inputs.reshape(-1, inputs.shape[-1])
        x = self.mlp_sdf(inputs)
        x = x.to(inputs)
        return x

    def sdf(self, x, training=True):
        return self.forward(x, training=training)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x,):
        with torch.set_grad_enabled(True):
            x = x + 0 * self.__hidden__(x)
            y = self.sdf(x)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        return gradients.reshape(-1, 3)


class RenderingNetwork(nn.Module):
    def __init__(
        self,
        geo_feat_dim     : int   = 128,
        d_hidden         : int   = 128,
        n_layers         : int   = 6,
        use_viewdirs     : bool  = True,
        mode             : str   = "no_xyz",
        n_frequencies    : int   = 4,
        device           : str   = 'cuda',
        **kwargs,
    ):
        super().__init__()

        self.kwargs = kwargs
        self.use_viewdirs = use_viewdirs
        self.mode = mode

        self.device = device

        if use_viewdirs:
            config_encoding_dirs = {
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": n_frequencies,
                    },
                ],
            }
            self.encoding_dirs = get_encoding(n_input_dims=3, config=config_encoding_dirs, device=self.device)
            self.d_in = self.encoding_dirs.n_output_dims

        self.mlp_head_in = 3 * (mode != "no_xyz") + self.d_in * use_viewdirs + 3 + geo_feat_dim  # points, viewdirs, normals, geo_feat
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",
            "n_neurons": d_hidden,
            "n_hidden_layers": n_layers,
        }
        self.mlp_head = get_mlp(
            n_input_dims=self.mlp_head_in, n_output_dims=3, config=network_config, device=self.device,
        )

        self.print_n_params("RenderingNetwork")

    def print_n_params(self, name):
        n_params = sum(p.numel() for p in self.parameters())
        n_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"* {name} * - #Params = {n_params}; #Trainable Params = {n_train_params}", flush=True)

    def forward(self, points, normals, view_dirs, feature_vectors,):
        view_dirs = (view_dirs + 1.) / 2. # (-1, 1) => (0, 1)
        if self.use_viewdirs:
            view_dirs = self.encoding_dirs(view_dirs)
        if self.mode == "no_xyz":
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        else:
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        x = self.mlp_head(rendering_input)
        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val: float):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    @property
    def inv_s(self):
        return torch.exp(self.variance * 10.0)

    def forward(self, x):
        return torch.ones([len(x), 1]) * self.inv_s
