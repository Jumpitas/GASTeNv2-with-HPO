# src/gan/architectures/stl10_stylegan2.py
from __future__ import annotations
import math, pickle, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

##############################################################################
# helpers
##############################################################################
def pixel_norm(x, eps=1e-8): return x * torch.rsqrt(torch.mean(x**2, 1, keepdim=True) + eps)

def he(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=.2)
        if m.bias is not None: nn.init.zeros_(m.bias)

##############################################################################
# mapping
##############################################################################
class Mapping(nn.Sequential):
    def __init__(self, z=128, w=512, n=8):
        super().__init__(*sum([[nn.Linear(z if i==0 else w, w), nn.LeakyReLU(.2)] for i in range(n)], []))
        self.apply(he)
    def forward(self, z): return pixel_norm(super().forward(z))

##############################################################################
# mod-conv
##############################################################################
class ModConv(nn.Module):
    def __init__(self, ic, oc, k=3, d=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(oc, ic, k, k))
        self.m = nn.Linear(512, ic)
        self.k, self.p, self.d = k, k//2, d
    def forward(self, x, s):
        b, c, h, w = x.shape
        style = self.m(s).view(b,1,c,1,1)+1
        w = self.w[None]*style
        if self.d:
            dem = torch.rsqrt((w**2).sum([2,3,4])+1e-8)
            w = w*dem.view(b,-1,1,1,1)
        x = x.view(1,-1,h,w)
        w = w.view(-1,c,self.k,self.k)
        return F.conv2d(x,w,padding=self.p,groups=b).view(b,-1,h,w)

##############################################################################
# generator block
##############################################################################
class GBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c1 = ModConv(ic, oc); self.c2 = ModConv(oc, oc)
        self.n1 = nn.Parameter(torch.zeros(1, oc, 1, 1))
        self.n2 = nn.Parameter(torch.zeros(1, oc, 1, 1))
        self.act = nn.LeakyReLU(.2)
    def forward(self, x, w):
        x = self.up(x)
        x = self.act(self.c1(x,w)+self.n1*torch.randn_like(x))
        x = self.act(self.c2(x,w)+self.n2*torch.randn_like(x))
        return x

##############################################################################
# discriminator block
##############################################################################
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

##############################################################################
# generator
##############################################################################
class Generator(nn.Module):
    def __init__(self, z_dim=128, img_size=(3,96,96), fmap=64):
        super().__init__()
        c,h,_ = img_size
        self.mapping = Mapping(z_dim)
        log = int(math.log2(h))
        self.n_blocks = log-2
        self.const = nn.Parameter(torch.randn(1,fmap*16,3,3))
        in_ch = fmap*16
        blocks, to_rgbs = [], []
        for i in range(self.n_blocks):
            out_ch = max(fmap, in_ch//2)
            blocks.append(GBlock(in_ch,out_ch))
            to_rgbs.append(ModConv(out_ch,c,1,False))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.torgb  = nn.ModuleList(to_rgbs)
        self.tanh = nn.Tanh()
        self.apply(he)
    def forward(self,z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0),-1,-1,-1)
        rgb = None
        for blk,tor in zip(self.blocks,self.torgb):
            x = blk(x,w)
            rgb = tor(x,w) if rgb is None else F.interpolate(rgb,scale_factor=2,mode='nearest')+tor(x,w)
        return self.tanh(rgb)

##############################################################################
# discriminator
##############################################################################
class Discriminator(nn.Module):
    def __init__(self, img_size=(3,96,96), fmap=64):
        super().__init__()
        c,h,_ = img_size
        log = int(math.log2(h))
        self.n_blocks = log-2
        layers=[spectral_norm(nn.Conv2d(c,fmap,3,1,1))]
        in_ch=fmap
        for _ in range(self.n_blocks):
            out_ch=min(fmap*16,in_ch*2)
            layers.append(DBlock(in_ch,out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.final = spectral_norm(nn.Linear(in_ch*3*3,1))
        self.apply(he)
    def forward(self,x):
        x = self.blocks(x)
        x = self.final(x.view(x.size(0),-1))
        return x.view(-1)

##############################################################################
# builders
##############################################################################
def build_stl10_g(z_dim=128, base_ch=64): return Generator(z_dim,(3,96,96),base_ch)
def build_stl10_d(base_ch=64):            return Discriminator((3,96,96),base_ch)
