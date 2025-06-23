# src/gan/architectures/stl10_sagan.py

"""
SAGAN for CIFAR-10 (3×32×32) and STL-10 (3×96×96)

* You can now call Generator/Discriminator with either
    - Generator(32, z_dim=..., filter_dim=..., n_blocks=...)    # -> RGB 32×32
    - Generator((3,96,96), z_dim=..., filter_dim=..., n_blocks=...)  # -> RGB 96×96
* No more “Use (C,H,W) tuple or single int for RGB.” errors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ────────────────────────────────────────────────────────────────────────────
# internal helper to parse “what size did you pass me?”
# ────────────────────────────────────────────────────────────────────────────
def _parse_size_and_channels(arg, img_ch: int | None = None):
    """
    If `arg` is an int → assume square RGB and return (3, arg, arg) (or img_ch if provided)
    If `arg` is a tuple/list of length 3 → return it directly as (C, H, W)
    """
    if isinstance(arg, int):
        C = img_ch or 3
        s = int(arg)
        return (C, s, s)
    if isinstance(arg, (tuple, list)) and len(arg) == 3:
        return (int(arg[0]), int(arg[1]), int(arg[2]))
    raise ValueError("Generator/Discriminator must be called with an int or a (C,H,W) tuple.")


# ────────────────────────────────────────────────────────────────────────────
# weight‐scaled conv, swish, init
# ────────────────────────────────────────────────────────────────────────────
class ScaledConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * (2 / self.weight[0].numel()) ** 0.5
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


# ────────────────────────────────────────────────────────────────────────────
# Self‐Attention (SAGAN)
# ────────────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch,   1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h*w)
        g = self.g(x).view(b, -1, h*w)
        beta = torch.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)  # B×HW×HW
        h_ = self.h(x).view(b, -1, h*w)
        o  = torch.bmm(h_, beta).view(b, c//2, h, w)
        return x + self.gamma * self.v(o)


# ────────────────────────────────────────────────────────────────────────────
# Residual up / down blocks
# ────────────────────────────────────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1    = ScaledConv2d(in_ch,  out_ch, 3,1,1)
        self.conv2    = ScaledConv2d(out_ch, out_ch, 3,1,1)
        self.skip     = ScaledConv2d(in_ch,  out_ch, 1,1,0)
        self.bn1      = nn.InstanceNorm2d(in_ch,  affine=True)
        self.bn2      = nn.InstanceNorm2d(out_ch, affine=True)
        self.act      = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.upsample(x)))
        y = self.conv1(y)
        y = self.act(self.bn2(y))
        y = self.conv2(y)
        skip = self.skip(self.upsample(x))
        return y + skip


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        Conv = (lambda *a, **k: spectral_norm(nn.Conv2d(*a,**k))) if use_sn else nn.Conv2d
        self.conv1 = Conv(in_ch,  out_ch, 3,1,1)
        self.conv2 = Conv(out_ch, out_ch, 3,1,1)
        self.skip  = Conv(in_ch,  out_ch, 1,1,0)
        self.pool  = nn.AvgPool2d(2)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y    = self.act(self.conv1(x))
        y    = self.act(self.conv2(y))
        y    = self.pool(y)
        skip = self.pool(self.skip(x))
        return y + skip


# ────────────────────────────────────────────────────────────────────────────
# Generator
# ────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self,
                 *args,
                 img_ch: int | None = None,
                 z_dim: int = 128,
                 filter_dim: int = 64,
                 n_blocks: int = 4):
        """
        G = Generator(32,            z_dim=128, filter_dim=64, n_blocks=3)  #→3×32×32
        G = Generator((3,96,96),     z_dim=128, filter_dim=64, n_blocks=4)  #→3×96×96
        """
        super().__init__()
        C, H, W = _parse_size_and_channels(args[0], img_ch=img_ch)

        # clamp number of up‐blocks so initial spatial ≥4x4
        max_b = max(int(math.log2(min(H, W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_b)
        self.z_dim    = z_dim

        self.init_h  = H // (2 ** self.n_blocks)
        self.init_w  = W // (2 ** self.n_blocks)
        self.init_ch = filter_dim * (2 ** self.n_blocks)

        self.project = nn.Sequential(
            nn.Linear(z_dim, self.init_ch * self.init_h * self.init_w),
            Swish()
        )

        blocks, ch = [], self.init_ch
        for i in range(self.n_blocks):
            nxt = ch // 2
            blocks.append(UpBlock(ch, nxt))
            ch = nxt
            if self.init_h * (2 ** (i+1)) in (32, 64):
                blocks.append(SelfAttention(ch))
        self.up_blocks = nn.Sequential(*blocks)

        self.to_rgb = nn.Sequential(
            ScaledConv2d(ch, C, 3,1,1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(1) != self.z_dim:
            raise ValueError(f"Expected noise dim {self.z_dim}, got {z.size(1)}")
        x = self.project(z).view(-1, self.init_ch, self.init_h, self.init_w)
        return self.to_rgb(self.up_blocks(x))


# ────────────────────────────────────────────────────────────────────────────
# Discriminator
# ────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self,
                 *args,
                 img_ch: int | None = None,
                 filter_dim: int = 64,
                 n_blocks: int = 4,
                 is_critic: bool = False):
        """
        D = Discriminator(32,            filter_dim=64, n_blocks=3, is_critic=False)
        D = Discriminator((3,96,96),     filter_dim=64, n_blocks=4, is_critic=True)
        """
        super().__init__()
        C, H, W = _parse_size_and_channels(args[0], img_ch=img_ch)

        max_b = max(int(math.log2(min(H, W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_b)

        ch = filter_dim
        blocks = [DownBlock(C, ch)]
        res = H // 2
        for _ in range(1, self.n_blocks):
            nxt = ch * 2
            blocks.append(DownBlock(ch, nxt))
            ch, res = nxt, res // 2
            if res in (64, 32):
                blocks.append(SelfAttention(ch))
        self.features = nn.Sequential(*blocks)

        # simple “global” head + “patch” head
        self.global_head = spectral_norm(nn.Conv2d(ch, 1, 1, 1, 0))
        self.is_critic   = is_critic

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)               # B×ch×H'×W'
        p = self.global_head(h)            # B×1×H'×W'
        glob  = p.mean([2,3])              # B×1
        patch = F.avg_pool2d(p, 4).mean([2,3])  # B×1
        out = glob + patch
        return out.view(-1) if self.is_critic else torch.sigmoid(out).view(-1)


# ────────────────────────────────────────────────────────────────────────────
# convenience functions
# ────────────────────────────────────────────────────────────────────────────
def build_cifar10_g(z_dim=128, base_ch=64):
    return Generator(32, img_ch=3, z_dim=z_dim, filter_dim=base_ch, n_blocks=3)

def build_cifar10_d(base_ch=64, critic=False):
    return Discriminator(32, img_ch=3, filter_dim=base_ch, n_blocks=3, is_critic=critic)

def build_stl10_g(z_dim=128, base_ch=64):
    return Generator((3, 96, 96), z_dim=z_dim, filter_dim=base_ch, n_blocks=4)

def build_stl10_d(base_ch=64, critic=False):
    return Discriminator((3, 96, 96), img_ch=3, filter_dim=base_ch, n_blocks=4, is_critic=critic)
