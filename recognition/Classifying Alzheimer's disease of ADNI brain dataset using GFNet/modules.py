import math
from functools import partial
import torch
import torch.nn as nn
import torch.fft
from timm.layers import DropPath, trunc_normal_  

class Mlp(nn.Module):
    """Two-layer MLP with GELU and dropout"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.4):
        super().__init__()
        out_features   = out_features   or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

#Global Filter 
class GlobalFilter(nn.Module):
    """Frequency-domain global filter via 2D FFT."""
    def __init__(self, h=14, w=8, dim=1000):
        super().__init__()
        # learnable complex weights (real, imag) last dim = 2
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.dim = dim
        self.h = h
        self.w = w

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.view(B, H, W, C)
        # forward FFT over spatial axes
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        w = torch.view_as_complex(self.complex_weight)
        x = x * w
        # inverse FFT to spatial
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        return x

# GF Block
class GFBlock(nn.Module):
    """Norm → GlobalFilter → Norm → MLP (+ DropPath residual)."""
    def __init__(self, dim, mlp_ratio=2., drop=0.5, drop_path=0.6,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim=dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = MLP(in_features=dim, hidden_features=hidden,
                               act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.ffn(self.norm2(self.filter(self.norm1(x)))))
        return x

# Patch projector
class PatchEmbed(nn.Module):
    """Image to patch embeddings via strided conv."""
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=512):
        super().__init__()
        self.img_size   = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, (
            f"Got {H}x{W}, expected {self.img_size[0]}x{self.img_size[1]}"
        )
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x

# GFNet main function
class GFNet(nn.Module):
    """
    GFNet backbone for AD vs NC (2 classes).
    Assumes 256×256 single-channel input grayscale MRI slices.
    """
    def __init__(self, img_size=256, patch_size=16, embed_dim=512, num_classes=2,
                 in_channels=1, drop_rate=0.5, depth=8, mlp_ratio=4.,
                 drop_path_rate=0.15, norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # positional embedding + dropout
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        # spatial sizes for FFT filter
        h_tokens = img_size // patch_size
        w_tokens = h_tokens // 2 + 1  # rfft width

        # per-block drop-path schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            GFBlock(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    h=h_tokens, w=w_tokens)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes)  # classifier

        self.apply(self._init_weights)

    # weight init mirrors  =
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # feature pipeline
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(1)  # global average over tokens
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
