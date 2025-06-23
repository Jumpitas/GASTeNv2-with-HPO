"""
SAGAN for CIFAR-10 (3×32×32) and STL-10 (3×96×96)
──────────────────────────────────────────────────
* Adaptive global/patch heads (no kernel-size crashes)
* Generator checks that the noise dimension matches `G.z_dim`
* Robust weight initialiser: skips InstanceNorm/BatchNorm when weight=None
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ────────────────────────────────────────────────────────────────────
# size helper
# ────────────────────────────────────────────────────────────────────
def _parse_size_and_channels(*args, img_ch=None, img_size=None):
    if args:
        first = args[0]
        if isinstance(first, (tuple, list)) and len(first) == 3:
            return map(int, first)
        if isinstance(first, int):            # legacy: square RGB
            s = int(first)
            return 3, s, s
        raise ValueError("Use (C,H,W) tuple or single int for RGB.")

    C = int(img_ch or 3)
    if img_size is None:
        raise ValueError("Need img_size=… when no positional size given")
    if isinstance(img_size, int):
        H = W = int(img_size)
    else:
        H, W = map(int, img_size)
    return C, H, W


# ────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────
class ScaledConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight * (2 / self.weight[0].numel()) ** 0.5
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        # skip if layer is affine-less
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ────────────────────────────────────────────────────────────────────
# attention and residual blocks
# ────────────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch,   1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h * w)
        g = self.g(x).view(b, -1, h * w)
        beta = torch.softmax(torch.bmm(f.permute(0, 2, 1), g), dim=-1)
        h_ = self.h(x).view(b, -1, h * w)
        o  = torch.bmm(h_, beta).view(b, c // 2, h, w)
        return x + self.gamma * self.v(o)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = ScaledConv2d(in_ch,  out_ch, 3,1,1)
        self.conv2 = ScaledConv2d(out_ch, out_ch, 3,1,1)
        self.skip  = ScaledConv2d(in_ch,  out_ch, 1,1,0)
        self.bn1, self.bn2 = nn.InstanceNorm2d(in_ch, True), nn.InstanceNorm2d(out_ch, True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.upsample(x)))
        y = self.conv1(y)
        y = self.act(self.bn2(y))
        y = self.conv2(y)
        return y + self.skip(self.upsample(x))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_sn=True):
        super().__init__()
        Conv = (lambda *a, **k: spectral_norm(nn.Conv2d(*a, **k))) if use_sn else nn.Conv2d
        self.conv1, self.conv2 = Conv(in_ch, out_ch, 3,1,1), Conv(out_ch, out_ch, 3,1,1)
        self.skip  = Conv(in_ch, out_ch, 1,1,0)
        self.down  = nn.AvgPool2d(2)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.down(y)
        return y + self.down(self.skip(x))


# ────────────────────────────────────────────────────────────────────
# generator
# ────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, *args, img_ch=None, img_size=None,
                 z_dim=128, filter_dim=64, n_blocks=4):
        super().__init__()
        C, H, W = _parse_size_and_channels(*args, img_ch=img_ch, img_size=img_size)

        self.z_dim = z_dim
        self.n_blocks = min(n_blocks, max(int(math.log2(min(H, W))) - 2, 1))

        self.init_h = H // (2 ** self.n_blocks)
        self.init_w = W // (2 ** self.n_blocks)
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
            if self.init_h * (2 ** (i + 1)) in (32, 64):
                blocks.append(SelfAttention(ch))
        self.up_blocks = nn.Sequential(*blocks)

        self.to_rgb = nn.Sequential(
            ScaledConv2d(ch, C, 3,1,1),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z):
        if z.size(1) != self.z_dim:
            raise ValueError(f"Generator expects z_dim={self.z_dim}, got {z.size(1)}")
        x = self.project(z).view(-1, self.init_ch, self.init_h, self.init_w)
        return self.to_rgb(self.up_blocks(x))


# ────────────────────────────────────────────────────────────────────
# discriminator
# ────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, *args, img_ch=None, img_size=None,
                 filter_dim=64, n_blocks=4, is_critic=False):
        super().__init__()
        C, H, W = _parse_size_and_channels(*args, img_ch=img_ch, img_size=img_size)

        self.n_blocks = min(n_blocks, max(int(math.log2(min(H, W))) - 2, 1))
        ch, blocks, res = filter_dim, [DownBlock(C, filter_dim)], H // 2

        for _ in range(1, self.n_blocks):
            nxt = ch * 2
            blocks.append(DownBlock(ch, nxt))
            ch, res = nxt, res // 2
            if res in (64, 32):
                blocks.append(SelfAttention(ch))
        self.features = nn.Sequential(*blocks)

        self.proj    = spectral_norm(nn.Conv2d(ch, 1, 1))
        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x):
        h = self.features(x)                 # B × ch × h × w
        proj = self.proj(h)                  # B × 1  × h × w
        glob  = proj.mean([2,3])
        patch = F.avg_pool2d(proj, kernel_size=min(4, h.shape[2])).mean([2,3])
        return self.act_out(glob + patch).view(-1)


# ────────────────────────────────────────────────────────────────────
# convenience builders
# ────────────────────────────────────────────────────────────────────
def build_cifar10_g(z_dim=128, base_ch=64):
    return Generator((3, 32, 32), z_dim=z_dim, filter_dim=base_ch, n_blocks=3)

def build_cifar10_d(base_ch=64, critic=False):
    return Discriminator((3, 32, 32), filter_dim=base_ch, n_blocks=3, is_critic=critic)

def build_stl10_g(z_dim=128, base_ch=64):
    return Generator((3, 96, 96), z_dim=z_dim, filter_dim=base_ch, n_blocks=4)

def build_stl10_d(base_ch=64, critic=False):
    return Discriminator((3, 96, 96), filter_dim=base_ch, n_blocks=4, is_critic=critic)
