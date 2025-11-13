# Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

# Third party
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

import torch.nn.functional as F
import math

from .components.cnn_blocks import (
    DownBlock,
    Downsample,
    MiddleBlock,
    UpBlock,
    Upsample, 
)
import numpy as np
from timm.models.vision_transformer import LayerScale, Attention
import os
from scipy.special import sph_harm



import numpy as np
@register("vit")
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        history,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x.shape = [B,C,H,W]
        x = self.patch_embed(x)
        # x.shape = [B,num_patches,embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.norm(x)
        return x

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,in_channels,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.head(x)
        # x.shape = [B,num_patches,embed_dim]
        preds = self.unpatchify(x)
        # preds.shape = [B,out_channels,H,W]
        return preds

@register("vitcc")
class VisionTransformerCC(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        history,
        oro_path,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels * history + 1   # original
        self.out_channels = out_channels
        self.patch_size = patch_size
        
        if oro_path:
            print("Loading orography from:", oro_path)
            # <<< ADDED: load and prepare orography
            oro = np.load(oro_path)["orography"]       # key from inspect
            oro = np.squeeze(oro).astype(np.float32)   # shape (534,534)
            oro = (oro - oro.min()) / (oro.max() - oro.min())  # normalize 0–1
            oro = torch.from_numpy(oro).unsqueeze(0)   # shape [1,H,W]
            self.register_buffer("oro", oro)           # static, no gradients
        else:
            print("No orography loaded.")
        # patch embedding now accounts for extra channel if ORO
        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        x = self.patch_embed(x)   # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    
    def forward(self, x):
        if len(x.shape) == 5:  # [B, T, in_channels, H, W]
            x = x.flatten(1, 2)  # [B, T*in_channels, H, W]

        B, _, H, W = x.shape
        
        # <<< ADDED: add orography channel
        oro = self.oro.unsqueeze(0).expand(B, -1, H, W)   # [B,1,H,W]
        x = torch.cat([x, oro], dim=1)                    # [B,C+1,H,W]

        # optional debug (only on first forward call)
        if not hasattr(self, "_printed"):
            print(f"Input with orography: {x.shape}")  # e.g. [B, T*in_channels+1, H, W]
            self._printed = True

        x = self.forward_encoder(x)
        x = self.head(x)
        preds = self.unpatchify(x)
        return preds

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

class Mapper_Vit(nn.Module):
    def __init__(
                self, 
                img_size,
                in_channels, # num_coefficients
                out_channels, # num_coefficients
                history=1,
                patch_size=16,
                drop_path=0.1,
                drop_rate=0.1,
                learn_pos_emb=False,
                embed_dim=1024,
                depth=24,
                decoder_depth=8,
                num_heads=16,
                mlp_ratio=4.0,
                history_mltpl=False,
                # activation="leaky"
                ):
        super().__init__()
        # if activation == "gelu":
        #     self.activation = nn.GELU()
        # elif activation == "relu":
        #     self.activation = nn.ReLU()
        # elif activation == "silu":
        #     self.activation = nn.SiLU()
        # elif activation == "leaky":
        #     self.activation = nn.LeakyReLU(0.3)
        # else:
        #     raise NotImplementedError(f"Activation {activation} not implemented")
        
        self.img_size = img_size
        self.in_channels = in_channels
        if history_mltpl:
            self.in_channels *= history
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)
        self.initialize_weights()
        

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x.shape = [B,C,H,W]
        x = self.patch_embed(x) ## HAVE torch.Size([16, 64, 534, 534])  Expect: 192 channels?
        # x.shape = [B,num_patches,embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.norm(x)
        return x

    def forward(self, A):  # [B, n_coeff, H, W]
        x = A
        if len(x.shape) == 5:  # x.shape = [B,T,in_channels,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.head(x)
        # x.shape = [B,num_patches,embed_dim]
        preds = self.unpatchify(x)
        # preds.shape = [B,out_channels,H,W]
        return preds  # [B, n_coeff, H, W]

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

class Geo_INR(nn.Module):
    def __init__(
            self,
            n_sh_coeff, 
            basis, 
            oro_path=None, 
            in_channels=1,
            conv_start_size=64,
            siren_hidden=128,
            slim = False,
            far = True
        ):  # [n, H, W]
        super().__init__()
        eps = 1e-6
        self.far = far
        self.slim = slim
        if isinstance(oro_path, str):
            oro = np.load(oro_path)['orography']
            oro = (oro - oro.mean()) / (oro.std() + eps)
            oro = torch.tensor(oro, dtype=torch.float32)
        else:  
            oro = oro_path.to(torch.float32)
            std = oro.std(unbiased=False).clamp_min(eps)
            oro = (oro - oro.mean()) / std
        self.register_buffer("oro", oro.unsqueeze(0))  # [1, H, W]
        self.register_buffer("basis", basis.unsqueeze(0))  # 
        self.oro_encoder = nn.Sequential(
            PeriodicConv2D(3, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            PeriodicConv2D(32, n_sh_coeff, kernel_size=3, padding=1),
        )
        self.siren = SirenNet(dim_in=n_sh_coeff, dim_hidden=siren_hidden, num_layers=2, dim_out=n_sh_coeff)
        self.conv = nn.Sequential(
            PeriodicConv2D(conv_start_size, 2*conv_start_size, kernel_size=3, padding=1),
            nn.SiLU(),
            PeriodicConv2D(2*conv_start_size, 4*conv_start_size, kernel_size=3, padding=1),
            nn.SiLU(),
            PeriodicConv2D(4*conv_start_size, 2*conv_start_size, kernel_size=3, padding=1),
            nn.SiLU(),
            PeriodicConv2D(2*conv_start_size, n_sh_coeff*in_channels, kernel_size=1, padding=0), # output 64 if before the backbone, output 1 channel if after the backbone
        ) if not slim else nn.Sequential(
            PeriodicConv2D(conv_start_size, 2*conv_start_size, kernel_size=3, padding=1),
            nn.SiLU(),   # new, remove two layers to reduce computation
            PeriodicConv2D(2*conv_start_size, n_sh_coeff, kernel_size=1, padding=0), # # new, remove (two layers, and the *) to reduce computation
        )
        proj_dim_mltp = in_channels if far else 1
        self.projection = PeriodicConv2D(n_sh_coeff*proj_dim_mltp, conv_start_size, kernel_size=1)
        if not far: self.projection_A = PeriodicConv2D(in_channels, n_sh_coeff, kernel_size=1)
        self.n_sh_coeff=n_sh_coeff
        self.in_channels = in_channels

    def forward(self, A):  # [B, n, H, W] (FAR) or [B, C, H, W] (noFar)
        B, _, H, W = A.size() 
        C = self.in_channels
        loc_basis = self.basis.view(-1, self.n_sh_coeff, H, W) # [1, 64, H, W]
        # Orography ––––––––––––––––––––––––––––––––
        oro = self.oro.view(-1, H, W)  # [B, H, W]
        oro = oro.unsqueeze(1)  # [B, 1, H, W]
        # Compute gradients ––––––––––––––––––––––––––––––––
        dx = oro[:, :, :, 1:] - oro[:, :, :, :-1]  # [B, 1, H, W-1]
        last_col = dx[:, :, :, -1:].clone()        # replicate last column
        grad_x = torch.cat([dx, last_col], dim=-1)  # [B, 1, H, W
        dy = oro[:, :, 1:, :] - oro[:, :, :-1, :]  # [B, 1, H-1, W]
        last_row = dy[:, :, -1:, :].clone()
        grad_y = torch.cat([dy, last_row], dim=-2)  # [B, 1, H, W]
        oro_feat = torch.cat([oro, grad_x, grad_y], dim=1)  # [B, 3, H, W]
        # Encode orography features ––––––––––––––––––––––––––––––––
        oro_basis = self.oro_encoder(oro_feat)  # [B, n_sh, H, W]
        geo_basis = loc_basis + oro_basis # [1, n_sh, H, W]
        geo_basis = self.siren(geo_basis.permute(0,2,3,1))#.view(self.n_sh_coeff, H, W)
        geo_basis = geo_basis.permute(0, 3, 1, 2) # [B, n_sh, H, W]
        ## ––––––––––––––––––––––––––––––––
        if self.far: ## have multiple channels form Far need to stack the geobasis
            geo_basis = geo_basis.repeat(1, C, 1, 1)
        else: 
            A = self.projection_A(A)  # project input A to n_sh_coeff channels
        fused = geo_basis * A  +  geo_basis
        fused = self.projection(fused)
        out = self.conv(fused)
        
        return out  # [B, 1, H, W]

@register("geofar")
class GeoFAR(nn.Module):
    def __init__(
            self, 
            img_size,
            in_channels, 
            out_channels, 
            history,  
            n_coeff=36,  ## <--- GEO PARAM
            n_sh_coeff=36,  ## <--- GEO PARAM
            conv_start_size=64, ## <--- embed arch param
            siren_hidden=128, ## <--- embed arch param
            patch_size=16, 
            drop_path=0.1,
            drop_rate=0.1,
            learn_pos_emb=False, 
            embed_dim=1024, 
            depth=24, 
            decoder_depth=8, 
            num_heads=16, 
            mlp_ratio=4.0,
            oro_path=None):

        super().__init__()
        H, W = img_size
        n_basis = int(math.sqrt(n_coeff))
        self.freqconv = LocalDCTConv(n=n_basis, k=8, in_channels=in_channels*history)
        self.basis = spherical_harmonic_basis(H, W, int(n_sh_coeff**0.5)) #.view(n_sh_coeff, H, W)
        self.encoder = Geo_INR(
            n_sh_coeff, 
            self.basis, 
            oro_path=oro_path, 
            in_channels=in_channels*history,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size
        ) 
        self.mapper_vit = Mapper_Vit(
            img_size=img_size, 
            in_channels=n_coeff, 
            out_channels=out_channels, 
            history=history, 
            patch_size=patch_size, 
            drop_path = drop_path,
            drop_rate = drop_rate,
            learn_pos_emb=learn_pos_emb, 
            embed_dim=embed_dim, 
            depth=depth, 
            decoder_depth=decoder_depth, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio, 
            history_mltpl=True,
        )

    def forward(self, I):  # I: [B, 1, h, w]
        I_freq = self.freqconv(I) # [B, n^2, H, W]
        I_freq = self.encoder(I_freq) # [B, n^2, H, W]
        I_prime = self.mapper_vit(I_freq)

        return I_prime

@register("geofar_v2")
class GeoFAR_v2(nn.Module):
    def __init__(
            self, 
            img_size,
            in_channels, 
            out_channels, 
            history,  
            n_coeff=36,  ## <--- GEO PARAM
            n_sh_coeff=36,  ## <--- GEO PARAM
            conv_start_size=64, ## <--- embed arch param
            siren_hidden=128, ## <--- embed arch param
            patch_size=16, 
            drop_path=0.1,
            drop_rate=0.1,
            learn_pos_emb=False, 
            embed_dim=1024, 
            depth=24, 
            decoder_depth=8, 
            num_heads=16, 
            mlp_ratio=4.0,
            oro_path=None):

        super().__init__()
        H, W = img_size
        n_basis = int(math.sqrt(n_coeff))
        self.freqconv = LocalDCTConv(n=n_basis, k=8, in_channels=in_channels*history)
        self.basis = spherical_harmonic_basis(H, W, int(n_sh_coeff**0.5)) #.view(n_sh_coeff, H, W)
        self.encoder = Geo_INR(
            n_sh_coeff, 
            self.basis, 
            oro_path=oro_path, 
            in_channels=in_channels*history,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size,
            slim=True,
            far=True
        ) 
        self.mapper_vit = Mapper_Vit(
            img_size=img_size, 
            in_channels=n_coeff, 
            out_channels=out_channels, 
            history=history, 
            patch_size=patch_size, 
            drop_path = drop_path,
            drop_rate = drop_rate,
            learn_pos_emb=learn_pos_emb, 
            embed_dim=embed_dim, 
            depth=depth, 
            decoder_depth=decoder_depth, 
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            history_mltpl=False,
        )

    def forward(self, I):  # I: [B, channels, h, w]
        #print(f'I.shape: {I.shape}') # torch.Size([16, 3, 1, 534, 534])
        I_last = I[:, -1, :, :]#.unsqueeze(1) # new
        #print(f'I_last.shape: {I_last.shape}') #  torch.Size([16, 1, 1, 534, 534])
        I_freq = self.freqconv(I) # [B, history*n^2, H, W]
        #print(f'I_freq.shape: after freqconv {I_freq.shape}') ## torch.Size([16, 192, 534, 534])
        I_freq = self.encoder(I_freq) # [B, n, H, W]
        #print(f'I_freq.shape: after encoder {I_freq.shape}') #torch.Size([16, 64, 534, 534])
        I_prime = self.mapper_vit(I_freq)
        #print(f'I_prime.shape: after mapping {I_prime.shape}') # torch.Size([16, 1, 534, 534])
        I = I_prime + I_last # new
        #print(f'I.shape: after mapping {I.shape}') # torch.Size([16, 16, 1, 534, 534])
        return I
 
@register("vitginr")
class VitGINR(nn.Module):
    def __init__(
            self,
            img_size,
            in_channels,
            out_channels,
            history,
            n_coeff=36,  ## <--- GEO PARAM
            n_sh_coeff=36,  ## <--- GEO PARAM
            conv_start_size=64, ## <--- embed arch param
            siren_hidden=128, ## <--- embed arch param
            patch_size=16,
            drop_path=0.1,
            drop_rate=0.1,
            learn_pos_emb=False,
            embed_dim=1024,
            depth=24,
            decoder_depth=8,
            num_heads=16,
            mlp_ratio=4.0,
            oro_path=None):
        super().__init__()
        H, W = img_size
        n_basis = int(math.sqrt(n_coeff))
        #self.freqconv = LocalDCTConv(n=n_basis, k=8, in_channels=in_channels*history)
        self.basis = spherical_harmonic_basis(H, W, int(n_sh_coeff**0.5)) #.view(n_sh_coeff, H, W)
        self.encoder = Geo_INR(
            n_sh_coeff,
            self.basis,
            oro_path=oro_path,
            in_channels=in_channels*history,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size,
            slim=True,
            far=False
        )
        self.mapper_vit = Mapper_Vit(
            img_size=img_size,
            in_channels=n_coeff,
            out_channels=out_channels,
            history=history,
            patch_size=patch_size,
            drop_path = drop_path,
            drop_rate = drop_rate,
            learn_pos_emb=learn_pos_emb,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            history_mltpl=False,
        )
    def forward(self, I):  # I: [B, c, h, w]
        #print(f'I.shape: {I.shape}') #torch.Size([16, 3, 1, 534, 534])
        I_last = I[:, -1, :, :] #.unsqueeze(1) # new
        #print(f'I_last.shape: {I_last.shape}') #torch.Size([16, 1, 534, 534])
        I_freq = I.squeeze(2)
        #print(f'I_freq.shape: {I_freq.shape}') #torch.Size([16, 3, 534, 534])
        I_freq = self.encoder(I_freq) # [B, n^2, H, W]
        #print(f'I_freq.shape: after encoder {I_freq.shape}')
        I_prime = self.mapper_vit(I_freq)
        #print(f'I_prime.shape: after mapping {I_prime.shape}')
        I = I_prime + I_last # new
        #print(f'I.shape: after mapping {I.shape}')
        return I 
