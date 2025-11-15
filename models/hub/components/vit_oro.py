import torch
import math
from torch import nn
import numpy as np
from scipy.special import sph_harm
import torch.nn.functional as F
def generate_local_dct_kernels(n: int, k: int) -> torch.Tensor:
    """
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.arange(k, dtype=torch.float32, device=device).view(-1, 1)
    y = torch.arange(k, dtype=torch.float32, device=device).view(1, -1)

    filters = []
    for u in range(n):
        for v in range(n):
            cu = math.sqrt(1.0 / k) if u == 0 else math.sqrt(2.0 / k)
            cv = math.sqrt(1.0 / k) if v == 0 else math.sqrt(2.0 / k)
            basis = cu * cv * torch.cos(math.pi * (2 * x + 1) * u / (2 * k)) * torch.cos(math.pi * (2 * y + 1) * v / (2 * k))
            filters.append(basis)
    
    filters = torch.stack(filters, dim=0).unsqueeze(1)  # [n*n, 1, k, k]
    return filters

def spherical_harmonic_basis(H, W, n_basis):
    """
    Generate [n_basis^2, H*W] real spherical harmonic basis sampled on an H x W grid.
    """
    lat_deg = np.linspace(-88.59375, 88.59375, H)  # shape: (H,), aligned with ERA5, while pseudo latitude coordinates for CERRA since spherical harmonics always require a global coverage
    lon_deg = np.linspace(0, 357.1875, W)          # shape: (W,), aligned with ERA5, while pseudo longtitude coordinates for CERRA
    lat_rad = np.radians(lat_deg)  # [-π/2, π/2]
    lon_rad = np.radians(lon_deg)  # [0, 2π)

    # Spherical coordinates
    theta = np.pi / 2 - lat_rad   # θ ∈ [0, π]
    phi = lon_rad
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")  # [H, W]

    # Compute harmonics Y_l^m over the grid
    basis_list = []
    for l in range(n_basis):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi_grid, theta_grid)  # complex array [H, W]
            Y_real = Y_lm.real  
            basis_list.append(Y_real.flatten())  # flatten to [H*W]

    B = np.stack(basis_list, axis=0)  # [n_basis², H*W]
    return torch.tensor(B, dtype=torch.float32)

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

class LocalDCTConv(nn.Module):
    def __init__(self, n=8, k=8, in_channels=1):
        super().__init__()
        filters = generate_local_dct_kernels(n, k)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels * (n*n),
                              kernel_size=k, stride=1, padding=k//2,
                              groups=in_channels)
        W = filters.repeat(in_channels, 1, 1, 1)        # [C*(n*n), 1, k, k]
        with torch.no_grad():
            self.conv.weight.copy_(W)
        self.conv.weight.requires_grad = False  # fixed filters

    def forward(self, x):
        x = x.squeeze(2)
        B, C, H, W = x.shape
        x = self.conv(x) # [B, C*(n*n), H, W]
        out = x[:,:,:H,:W]
        return out 

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, dropout = False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, degreeinput = False, dropout = False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            ))

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

        return self.last_layer(x)
