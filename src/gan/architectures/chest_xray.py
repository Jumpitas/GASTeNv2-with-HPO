from __future__ import annotations
import math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
class ScaledConv2d(nn.Conv2d):
    """He-style weight scaling à-la StyleGAN (works better with Leaky-ReLU)."""
    def forward(self, x):
        w = self.weight * (2 / self.weight[0].numel())**0.5
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Swish(nn.Module):
    def forward(self, x):                       # SiLU a.k.a. Swish
        return x * torch.sigmoid(x)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# ────────────────────────────────────────────────────────────────────
# Self-Attention block (SAGAN)
# ────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch,   1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h * w)                     # B×C/8×HW
        g = self.g(x).view(b, -1, h * w)                     # B×C/8×HW
        beta = torch.softmax(torch.bmm(f.permute(0, 2, 1), g), dim=-1)  # B×HW×HW
        h_ = self.h(x).view(b, -1, h * w)                    # B×C/2×HW
        o = torch.bmm(h_, beta).view(b, c // 2, h, w)        # B×C/2×H×W
        o = self.v(o)                                        # B×C×H×W
        return x + self.gamma * o

# ────────────────────────────────────────────────────────────────────
# Residual up / down blocks
# ────────────────────────────────────────────────────────────────────
class UpBlock(nn.Module):
    """Nearest-neighbour up-sample + residual 3×3 convs."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = ScaledConv2d(in_ch,  out_ch, 3, 1, 1)
        self.conv2 = ScaledConv2d(out_ch, out_ch, 3, 1, 1)
        self.skip  = ScaledConv2d(in_ch,  out_ch, 1, 1, 0)
        self.bn1 = nn.InstanceNorm2d(in_ch,  affine=True)
        self.bn2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.upsample(x)))
        y = self.conv1(y)
        y = self.act(self.bn2(y))
        y = self.conv2(y)
        skip = self.skip(self.upsample(x))
        return y + skip

class DownBlock(nn.Module):
    """Residual down-sample 3×3 convs (PatchGAN style)."""
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        Conv = (lambda *a, **k: spectral_norm(nn.Conv2d(*a, **k))
                if use_sn else nn.Conv2d)
        self.conv1 = Conv(in_ch, out_ch, 3, 1, 1)
        self.conv2 = Conv(out_ch, out_ch, 3, 1, 1)
        self.skip  = Conv(in_ch, out_ch, 1, 1, 0)
        self.down  = nn.AvgPool2d(2)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.down(y)
        skip = self.down(self.skip(x))
        return y + skip

# ────────────────────────────────────────────────────────────────────
# Generator
# ────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self,
                 image_size: tuple[int, int, int] = (1, 128, 128),
                 z_dim: int = 128,
                 filter_dim: int = 64,
                 n_blocks: int = 4):
        """
        image_size : (C, H, W) – supports resolutions 32 → 512.
                     For CXRs we fix to 1×128×128  (n_blocks ≥ 4).
        """
        super().__init__()
        c, h, w = image_size
        max_blocks = max(int(math.log2(min(h, w))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)
        self.z_dim = z_dim

        self.init_h = h // (2 ** self.n_blocks)
        self.init_w = w // (2 ** self.n_blocks)
        self.init_ch = filter_dim * 2 ** self.n_blocks

        self.project = nn.Sequential(
            nn.Linear(z_dim, self.init_ch * self.init_h * self.init_w),
            Swish()
        )

        blocks = []
        ch = self.init_ch
        for i in range(self.n_blocks):
            nxt = ch // 2
            blocks.append(UpBlock(ch, nxt))
            ch = nxt
            if self.init_h * 2 ** (i + 1) in (32, 64):
                blocks.append(SelfAttention(ch))
        self.up_blocks = nn.Sequential(*blocks)

        self.to_rgb = nn.Sequential(
            ScaledConv2d(ch, c, 3, 1, 1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, self.init_ch, self.init_h, self.init_w)
        x = self.up_blocks(x)
        return self.to_rgb(x)

# ────────────────────────────────────────────────────────────────────
# Discriminator
# ────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self,
                 image_size: tuple[int, int, int] = (1, 128, 128),
                 filter_dim: int = 64,
                 n_blocks: int = 4,
                 is_critic: bool = False):
        """
        PatchGAN + global head for better texture & silhouette.
        """
        super().__init__()
        c, h, w = image_size
        max_blocks = max(int(math.log2(min(h, w))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)

        ch = filter_dim
        blocks = [DownBlock(c, ch)]
        res = h // 2
        for i in range(1, self.n_blocks):
            nxt = ch * 2
            blocks.append(DownBlock(ch, nxt))
            ch = nxt
            res //= 2
            if res in (64, 32):
                blocks.append(SelfAttention(ch))
        self.features = nn.Sequential(*blocks)            # B×ch×res×res

        # global head
        self.global_head = spectral_norm(
            nn.Conv2d(ch, 1, res, 1, 0)
        )
        # 70×70 PatchGAN head
        self.patch_head = spectral_norm(
            nn.Conv2d(ch, 1, 4, 1, 0)
        )
        self.sigmoid = nn.Sigmoid() if not is_critic else nn.Identity()
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        glob = self.global_head(h).view(x.size(0), -1)     # B×1
        patch = self.patch_head(h).mean([2, 3])            # average over H,W
        out = glob + patch
        return self.sigmoid(out).view(-1)

# ────────────────────────────────────────────────────────────────────
# convenient constructors for CXRs
# ────────────────────────────────────────────────────────────────────
def build_cxr_g(z_dim: int = 128, base_ch: int = 64) -> Generator:
    return Generator((1, 128, 128), z_dim=z_dim, filter_dim=base_ch, n_blocks=4)

def build_cxr_d(base_ch: int = 64, critic: bool = False) -> Discriminator:
    return Discriminator((1, 128, 128), filter_dim=base_ch,
                         n_blocks=4, is_critic=critic)