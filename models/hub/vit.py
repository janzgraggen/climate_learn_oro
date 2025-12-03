# Local application
from .components.cnn_blocks import PeriodicConv2D
from .components.pos_embed import get_2d_sincos_pos_embed
from .utils import register

# Third party
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_
import math

import numpy as np

from .components.vit_oro import (
    spherical_harmonic_basis,
    LocalDCTConv,
    GeoINR
)

import numpy as np

@register("vit")
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
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
        history_mltpl=True,
    ):
        super().__init__()
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

    def forward(self, x):  # [B, n_coeff, H, W]
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

@register("vitcc2vit")
class VisionTransformerCC2vit(nn.Module):
    """
    Post-fusion ViT model using two Mapper_Vit models:
      - mapper_temp: receives temp-history
      - mapper_oro : receives orography (tiled or single)
    Fused by 1x1 conv.
    """

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
        H, W = img_size
        # Load & register orography
        oro = np.load(oro_path)["orography"].astype(np.float32)
        oro = (oro - oro.min()) / (oro.max() - oro.min())
        oro = torch.from_numpy(oro)[None, ...]  # [1,H,W]
        self.register_buffer("oro", oro)

        # mapper for temporal stack
        self.mapper_temp = VisionTransformer(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            history=history,
            patch_size=patch_size,
            drop_path=drop_path,
            drop_rate=drop_rate,
            learn_pos_emb=learn_pos_emb,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            history_mltpl=True,
        )
        # mapper for orography
        self.mapper_oro = VisionTransformer(
            img_size=img_size,
            in_channels=1,
            out_channels=out_channels,
            history=1,
            patch_size=patch_size,
            drop_path=drop_path,
            drop_rate=drop_rate,
            learn_pos_emb=learn_pos_emb,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            history_mltpl=False,
        )

        # Fusion head
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # TEMP BRANCH
        temp_pred = self.mapper_temp(x)  # [B,C,H,W]

        # ORO BRANCH
        B, _, _, H, W = x.shape
        oro = self.oro.expand(B, 1, H, W)       # → [B,1,H,W]
        oro_pred = self.mapper_oro(oro)                  # [B,C,H,W]

        # Fuse predictions
        fused = torch.cat([temp_pred, oro_pred], dim=1)  # [B,2C,H,W]
        out = self.fuse(fused)                           # [B,C,H,W]
        return out

@register("vitfuse")
class VisionTransformerFuse(nn.Module):
    """
    - replaces simple oro channel concatenation with fused oro
      x_fused = temp * oro_proj + oro_proj
    """

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
        self.in_channels = in_channels * history  # MODIFIED: no +1 channel
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

        # MODIFIED FOR vitFuse: projection conv for fusion
        self.oro_proj = nn.Conv2d(3, self.in_channels, kernel_size=1, bias=False)
        # MODIFIED FOR vitFuse: end

        # patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm,
                  proj_drop=drop_rate, attn_drop=drop_rate)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # head
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()

    # SAME AS vitCC
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
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward_encoder(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


    def forward(self, x):
        if len(x.shape) == 5:  # [B,T,C,H,W]
            x = x.flatten(1, 2)  # [B, C*history, H, W]

        B, C, H, W = x.shape
        
        #  <<< ADDED: add orography channel
        # oro = self.oro.unsqueeze(0).expand(B, -1, H, W)   # [B,1,H,W]
        # x = torch.cat([x, oro], dim=1)                    # [B,C+1,H,W]

        # compute oro gradients and stack
        oro = self.oro.unsqueeze(0).expand(B, -1, H, W)
        dx = oro[:, :, :, 1:] - oro[:, :, :, :-1]  # [B, 1, H, W-1]
        last_col = dx[:, :, :, -1:].clone()        # replicate last column
        dx = torch.cat([dx, last_col], dim=-1)  # [B, 1, H, W
        dy = oro[:, :, 1:, :] - oro[:, :, :-1, :]  # [B, 1, H-1, W]
        last_row = dy[:, :, -1:, :].clone()
        dy = torch.cat([dy, last_row], dim=-2)  # [B, 1, H, W]
        oro3 = torch.cat([oro, dx, dy], dim=1)  # [B,3,H,W]

        # fusion with projection conv to match x channels
        oro_proj = self.oro_proj(oro3)           # [B,C,H,W]
        x = x * oro_proj + oro_proj               # fused input

        # forward through ViT
        x = self.forward_encoder(x)
        x = self.head(x)
        preds = self.unpatchify(x)
        return preds
    # MODIFIED FOR vitFuse: end

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
        self.encoder = GeoINR(
            n_sh_coeff, 
            self.basis, 
            oro_path=oro_path, 
            in_channels=in_channels*history,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size
        ) 
        self.mapper_vit = VisionTransformer(
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
        self.encoder = GeoINR(
            n_sh_coeff, 
            self.basis, 
            oro_path=oro_path, 
            in_channels=in_channels*history,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size,
            slim=True,
            far=True
        ) 
        self.mapper_vit = VisionTransformer(
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
            n_sh_coeff=36,  ## <--- GEO PARAM  for special harmonics !!  REMOVED: #n_coeff=36,  ## <--- GEO PARAM for LocalDCTConvs
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
        self.basis = spherical_harmonic_basis(H, W, int(n_sh_coeff**0.5)) #.view(n_sh_coeff, H, W) !!! REMOVED: #n_basis = int(math.sqrt(n_coeff)), #self.freqconv = LocalDCTConv(n=n_basis, k=8, in_channels=in_channels*history)
        self.encoder = GeoINR(
            n_sh_coeff,
            self.basis,
            oro_path=oro_path,
            in_channels=in_channels*history,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size,
            slim=True,
            far=False
        )
        self.mapper_vit = VisionTransformer(
            img_size=img_size,
            in_channels=n_sh_coeff,
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
        I = I.squeeze(2)
        #print(f'I.shape after squeze(2): {I.shape}') #torch.Size([16, 3, 534, 534])
        I_enc = self.encoder(I) # [B, n^2, H, W]
        #print(f'I_enc.shape: after encoder {I_enc.shape}')
        I_prime = self.mapper_vit(I_enc)
        #print(f'I_prime.shape: after mapping {I_prime.shape}')
        I = I_prime + I_last # new
        #print(f'I.shape: after mapping (at the end) {I.shape}')
        return I 