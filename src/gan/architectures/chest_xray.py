# style_cxr_gan.py
from __future__ import annotations
import math, random, functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

###############################################################################
# Utils
###############################################################################
def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

def he_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if m.bias is not None: nn.init.zeros_(m.bias)

###############################################################################
# Mapping network (8-layer MLP)
###############################################################################
class Mapping(nn.Sequential):
    def __init__(self, z_dim=128, w_dim=512, n=8):
        layers = []
        for _ in range(n):
            layers += [nn.Linear(z_dim if not layers else w_dim, w_dim), nn.LeakyReLU(0.2)]
        super().__init__(*layers)
        self.apply(he_init)

    def forward(self, z): return pixel_norm(super().forward(z))

###############################################################################
# Modulated convolution (StyleGAN2)
###############################################################################
class ModConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, demod=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.mod    = nn.Linear(512, in_ch, bias=True)
        self.demod  = demod
        self.k      = k
        self.pad    = k//2

    def forward(self, x, w):
        b, c, h, w_ = x.shape
        s = self.mod(w).view(b, 1, c, 1, 1) + 1      # style
        weight = self.weight[None] * s
        if self.demod:
            d = torch.rsqrt((weight**2).sum([2,3,4])+1e-8)
            weight = weight * d.view(b, -1, 1, 1, 1)
        x = x.view(1, -1, h, w_)
        wgt = weight.view(-1, c, self.k, self.k)
        x = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return x.view(b, -1, h, w_)

###############################################################################
# Generator block
###############################################################################
class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = ModConv(in_ch, out_ch)
        self.conv2 = ModConv(out_ch, out_ch)
        self.noise1 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.noise2 = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.conv1(x, w) + self.noise1 * torch.randn_like(x)
        x = self.act(x)
        x = self.conv2(x, w) + self.noise2 * torch.randn_like(x)
        x = self.act(x)
        return x

###############################################################################
# StyleGAN-like Generator
###############################################################################
class Generator(nn.Module):
    def __init__(self, img_size=(1,128,128)):
        super().__init__()
        c, h, _ = img_size
        self.mapping = Mapping()
        fmap_base = 64
        log_size  = int(math.log2(h))
        self.const = nn.Parameter(torch.randn(1, fmap_base*16, 4, 4))

        blocks, to_rgbs = [], []
        in_ch = fmap_base*16
        for i in range(3, log_size+1):        # 8×8 … 128×128
            out_ch = max(fmap_base*16 // 2**(i-3), fmap_base)
            blocks.append(GBlock(in_ch, out_ch))
            to_rgbs.append(ModConv(out_ch, c, k=1, demod=False))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.to_rgbs= nn.ModuleList(to_rgbs)
        self.tanh   = nn.Tanh()
        self.apply(he_init)

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        out = None
        for blk, rgb in zip(self.blocks, self.to_rgbs):
            x = blk(x, w)
            out = rgb(x, w) if out is None else self.upsample(out) + rgb(x, w)
        return self.tanh(out)

###############################################################################
# Discriminator block (StyleGAN2 ResNet + SpectralNorm)
###############################################################################
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.down  = nn.AvgPool2d(2)
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        self.act   = nn.LeakyReLU(0.2)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.down(y)
        return y + self.down(self.skip(x))

###############################################################################
# Discriminator
###############################################################################
class Discriminator(nn.Module):
    def __init__(self, img_size=(1,128,128)):
        super().__init__()
        c, h, _ = img_size
        fmap_base = 64
        log_size  = int(math.log2(h))

        blocks = [spectral_norm(nn.Conv2d(c, fmap_base, 3, 1, 1))]
        in_ch = fmap_base
        for i in range(3, log_size+1):
            out_ch = min(fmap_base*16, in_ch*2)
            blocks.append(DBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.final_conv = spectral_norm(nn.Conv2d(in_ch, in_ch, 3, 1, 1))
        self.final_dense= spectral_norm(nn.Linear(in_ch*4*4, 1))

    def forward(self, x):
        x = self.blocks(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        return self.final_dense(x).view(-1)

###############################################################################
# Convenience builders for your training script
###############################################################################
def build_cxr_g(z_dim=128): return Generator((1,128,128))
def build_cxr_d():           return Discriminator((1,128,128))
