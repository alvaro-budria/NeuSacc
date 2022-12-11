import math
from typing import Any, Dict, Union
import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class VanillaMLP(nn.Module):
    def __init__(
        self,
        dim_in             : int,
        dim_out            : int,
        n_neurons          : int,
        n_hidden_layers    : int,
        sphere_init        : bool  = True,
        sphere_init_radius : float = 0.5,
        weight_norm        : bool  = True,
        output_activation  : str   = "Identity",
        **kwargs,
    ):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers
        self.sphere_init, self.weight_norm = sphere_init, weight_norm
        self.sphere_init_radius = sphere_init_radius
        self.layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [
                self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False),
                self.make_activation()
            ]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = getattr(nn, output_activation)()

    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in: int, dim_out: int, is_first: bool, is_last: bool):
        layer = nn.Linear(dim_in, dim_out, bias=True)
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


class VanillaFrequency(nn.Module):
    def __init__(self, in_channels: int, config: Dict[str, Any]):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get('n_masking_step', 0)
        self.update_step(None, None) # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq*x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch: int, global_step: int):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
            print("Initializing VanillaFrequency mask: ", self.mask)
        else:
            self.mask = (1. - torch.cos(math.pi * (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs)).clamp(0, 1))) / 2.
            print(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')


class CompositeEncoding(nn.Module):
    def __init__(
        self,
        encoding,
        include_xyz: bool = False,
        xyz_scale: float = 1.,
        xyz_offset: float = 0.,
        n_frequencies: int = 0,
        device: str = 'cuda',
    ):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = self.encoding.n_output_dims
        self.n_frequencies = n_frequencies
        self.device = device
        if self.include_xyz:
            self.n_output_dims += self.encoding.n_input_dims
        if n_frequencies > 0:
            config_encoding_x = {
                "otype": "VanillaFrequency",
                "n_frequencies": self.n_frequencies,
                "is_composite": False,
            }
            self.encoding_freq = get_encoding(self.encoding.n_input_dims, config_encoding_x, self.device,)
            self.n_output_dims += self.encoding_freq.n_output_dims

    def forward(self, x, *args):
        enc = self.encoding(x, *args)
        if self.n_frequencies > 0:
            enc = torch.cat([self.encoding_freq(x, *args), enc], dim=-1)
        if self.include_xyz:
            enc = torch.cat([self.xyz_scale * x + self.xyz_offset, enc], dim=-1)
        return enc


def get_encoding(n_input_dims: int, config: Dict[str, Any], device: Union[str, int]):
    # input is assumed to be in range [0, 1]
    if config.get("otype") == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, config)
    else:
        with torch.cuda.device(device):
            encoding = tcnn.Encoding(n_input_dims, config)
    if config.get("is_composite", True):
        encoding = CompositeEncoding(
            encoding,
            include_xyz=config.get('include_xyz', False),
            xyz_scale=2.,
            xyz_offset=-1.,
            n_frequencies=config.get('n_frequencies', 0),
            device=device,
        )
    return encoding


def sphere_init_tcnn_network(n_input_dims: int, n_output_dims: int, config: Dict[str, Any], network):
    print('Initialize tcnn MLP to approximately represent a sphere.')
    """
    from https://github.com/NVlabs/tiny-cuda-nn/issues/96
    It's the weight matrices of each layer laid out in row-major order and then concatenated.
    Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
    The padded input dimensions get a constant value of 1.0,
    whereas the padded output dimensions are simply ignored,
    so the weights pertaining to those can have any value.
    """
    padto = 16 if config.otype == 'FullyFusedMLP' else 8
    n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
    n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    data = list(network.parameters())[0].data
    assert data.shape[0] == (n_input_dims + n_output_dims) * config.n_neurons + (config.n_hidden_layers - 1) * config.n_neurons**2
    new_data = []
    # first layer
    weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
    torch.nn.init.constant_(weight[:, 3:], 0.0)
    torch.nn.init.normal_(weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
    new_data.append(weight.flatten())
    # hidden layers
    for i in range(config.n_hidden_layers - 1):
        weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
        torch.nn.init.normal_(weight, 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
        new_data.append(weight.flatten())
    # last layer
    weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
    torch.nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(config.n_neurons), std=0.0001)
    new_data.append(weight.flatten())
    new_data = torch.cat(new_data)
    data.copy_(new_data)


def get_mlp(n_input_dims: int, n_output_dims: int, config: Dict[str, Any], device: Union[str, int]):
    if config.get("otype") == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, **config)
    else:
        with torch.cuda.device(device):
            network = tcnn.Network(n_input_dims, n_output_dims, config)
            if config.get('sphere_init', False):
                sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    return network
