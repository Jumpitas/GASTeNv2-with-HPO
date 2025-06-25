from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ───── helpers ─────
def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(torch.mean(x**2, 1, keepdim=True) + eps)

def he(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ───── mapping ─────
class Mapping(nn.Sequential):
    def __init__(self, z=128, w=512, n=8):
        layers = []
        for i in range(n):
            in_f = z if i == 0 else w
            layers += [nn.Linear(in_f, w), nn.LeakyReLU(.2)]
        super().__init__(*layers)
        self.apply(he)
    def forward(self, z): return pixel_norm(super().forward(z))

# ───── mod-conv ─────
class ModConv(nn.Module):
    def __init__(self, ic, oc, k=3, demod=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(oc, ic, k, k))
        self.affine = nn.Linear(512, ic)
        self.k, self.pad, self.demod = k, k//2, demod
    def forward(self, x, w):
        b, c, h, w_ = x.shape
        style = self.affine(w).view(b,1,c,1,1) + 1
        wgt   = self.weight[None] * style
        if self.demod:
            d = torch.rsqrt((wgt**2).sum([2,3,4]) + 1e-8)
            wgt = wgt * d.view(b,-1,1,1,1)
        wgt = wgt.view(-1,c,self.k,self.k)
        x   = x.view(1,-1,h,w_)
        x   = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return x.view(b,-1,h,w_)

# ───── G-block ─────
class GBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1, self.c2 = ModConv(ic, oc), ModConv(oc, oc)
        self.n1 = nn.Parameter(torch.zeros(1, oc, 1, 1))
        self.n2 = nn.Parameter(torch.zeros(1, oc, 1, 1))
        self.act = nn.LeakyReLU(.2)
    def forward(self, x, w):
        x = self.up(x)
        x = self.act(self.c1(x, w) + self.n1*torch.randn_like(x))
        x = self.act(self.c2(x, w) + self.n2*torch.randn_like(x))
        return x

# ───── D-block ─────
class DBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv2d(ic, oc, 3,1,1))
        self.c2 = spectral_norm(nn.Conv2d(oc, oc, 3,1,1))
        self.skip= spectral_norm(nn.Conv2d(ic, oc, 1))
        self.pool= nn.AvgPool2d(2)
        self.act = nn.LeakyReLU(.2)
    def forward(self, x):
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))

# ───── Generator ─────
class Generator(nn.Module):
    def __init__(self,
                 z_dim: int = 128,
                 img_size: tuple[int,int,int] = (1,128,128),
                 fmap: int = 64,
                 **_):                       # **_ absorbs unused kwargs
        super().__init__()
        c,h,_ = img_size
        self.mapping = Mapping(z_dim)
        log = int(math.log2(h))             # 7 for 128
        self.const = nn.Parameter(torch.randn(1, fmap*16, 4, 4))

        in_ch = fmap*16
        blocks, to_rgbs = [], []
        for _ in range(log-2):              # 5 up-blocks → 128
            out_ch = max(fmap, in_ch//2)
            blocks.append(GBlock(in_ch, out_ch))
            to_rgbs.append(ModConv(out_ch, c, 1, demod=False))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.torgb  = nn.ModuleList(to_rgbs)
        self.tanh   = nn.Tanh()
        self.apply(he)

    def forward(self, z):
        w   = self.mapping(z)
        x   = self.const.expand(z.size(0), -1, -1, -1)
        rgb = None
        for blk, tor in zip(self.blocks, self.torgb):
            x   = blk(x, w)
            rgb = tor(x, w) if rgb is None else \
                  F.interpolate(rgb, scale_factor=2, mode="nearest") + tor(x, w)
        return self.tanh(rgb)

# ───── Discriminator ─────
class Discriminator(nn.Module):
    def __init__(self,
                 img_size: tuple[int,int,int] = (1,128,128),
                 fmap: int = 64,
                 **_):
        super().__init__()
        c,h,_ = img_size
        log = int(math.log2(h))
        layers=[spectral_norm(nn.Conv2d(c,fmap,3,1,1))]
        in_ch = fmap
        for _ in range(log-2):
            out_ch = min(fmap*16, in_ch*2)
            layers.append(DBlock(in_ch,out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.final  = spectral_norm(nn.Linear(in_ch*4*4, 1))
        self.apply(he)

    def forward(self, x):
        x = self.blocks(x)
        return self.final(x.flatten(1)).view(-1)

# ───── public builders ─────
def build_cxr_g(z_dim=128, base_ch=64):
    return Generator(z_dim=z_dim, img_size=(1,128,128), fmap=base_ch)

def build_cxr_d(base_ch=64):
    return Discriminator(img_size=(1,128,128), fmap=base_ch)
