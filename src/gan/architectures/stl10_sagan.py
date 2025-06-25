# src/gan/architectures/stl10_sagan.py
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

def pixel_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

def he(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Mapping(nn.Sequential):
    def __init__(self, z_dim=128, w_dim=512, layers=8):
        mods = []
        for i in range(layers):
            mods += [nn.Linear(z_dim if i == 0 else w_dim, w_dim), nn.LeakyReLU(0.2)]
        super().__init__(*mods)
        self.apply(he)
    def forward(self, z): return pixel_norm(super().forward(z))

class ModConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, demod=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.affine = nn.Linear(512, in_ch)
        self.pad, self.demod = k//2, demod
    def forward(self, x, w):
        b,c,h,w_ = x.shape
        style = self.affine(w).view(b,1,c,1,1) + 1
        wgt   = self.weight[None] * style
        if self.demod:
            d = torch.rsqrt((wgt**2).sum([2,3,4]) + 1e-8)
            wgt = wgt * d.view(b,-1,1,1,1)
        x   = x.view(1, -1, h, w_)
        wgt = wgt.view(-1, c, self.weight.size(2), self.weight.size(3))
        out = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return out.view(b, -1, h, w_)

class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up  = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1  = ModConv(in_ch,  out_ch)
        self.c2  = ModConv(out_ch, out_ch)
        self.n1  = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.n2  = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act = nn.LeakyReLU(0.2)
    def forward(self,x,w):
        x = self.up(x)
        x = self.act(self.c1(x,w) + self.n1 * torch.randn_like(x))
        x = self.act(self.c2(x,w) + self.n2 * torch.randn_like(x))
        return x

class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1   = spectral_norm(nn.Conv2d(in_ch,  out_ch, 3,1,1))
        self.c2   = spectral_norm(nn.Conv2d(out_ch, out_ch, 3,1,1))
        self.skip = spectral_norm(nn.Conv2d(in_ch,  out_ch, 1))
        self.pool = nn.AvgPool2d(2)
        self.act  = nn.LeakyReLU(0.2)
    def forward(self,x):
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))

class Generator(nn.Module):
    def __init__(self, img_size=(3,96,96), z_dim=128, fmap=64, n_blocks=4, **_):
        super().__init__()
        c,h,_ = img_size
        max_b = max(int(math.log2(h)) - 2, 1)
        self.n_blocks = min(n_blocks, max_b)
        init_h  = h // (2**self.n_blocks)
        init_ch = fmap * (2**self.n_blocks)
        self.z_dim   = z_dim
        self.mapping = Mapping(z_dim)
        self.const   = nn.Parameter(torch.randn(1, init_ch, init_h, init_h))
        self.ups     = nn.Upsample(scale_factor=2, mode="nearest")

        in_ch = init_ch
        self.blocks = nn.ModuleList()
        self.torgb  = nn.ModuleList()
        for _ in range(self.n_blocks):
            out_ch = max(fmap, in_ch//2)
            self.blocks.append(GBlock(in_ch, out_ch))
            self.torgb.append(ModConv(out_ch, c, 1, demod=False))
            in_ch = out_ch
        self.tanh = nn.Tanh()
        self.apply(he)

    def forward(self,z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        rgb = None
        for blk, tor in zip(self.blocks, self.torgb):
            x   = blk(x,w)
            rgb = tor(x,w) if rgb is None else self.ups(rgb) + tor(x,w)
        return self.tanh(rgb)

class Discriminator(nn.Module):
    def __init__(self, img_size=(3,96,96), fmap=64, n_blocks=4, is_critic=False, **_):
        super().__init__()
        c,h,_ = img_size
        max_b = max(int(math.log2(h)) // 2, 1)
        self.n_blocks = min(n_blocks, max_b)
        res    = h // (2**self.n_blocks)
        layers = [spectral_norm(nn.Conv2d(c, fmap, 3,1,1))]
        in_ch  = fmap
        for _ in range(self.n_blocks):
            out_ch = min(fmap*16, in_ch*2)
            layers.append(DBlock(in_ch, out_ch))
            in_ch = out_ch
            res  //= 2
        res = max(res, 1)
        self.features = nn.Sequential(*layers)
        self.final_fc = spectral_norm(nn.Linear(in_ch*res*res, 1))
        self.out_act  = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(he)

    def forward(self,x):
        h = self.features(x)
        return self.out_act(self.final_fc(h.flatten(1))).view(-1)

def build_stl10_g(z_dim=128, base_ch=64):
    return Generator((3,96,96), z_dim=z_dim, fmap=base_ch)

def build_stl10_d(base_ch=64, critic=False):
    return Discriminator((3,96,96), fmap=base_ch, is_critic=critic)
