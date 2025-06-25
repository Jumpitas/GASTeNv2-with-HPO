from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ───────── helpers ─────────
def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

def he_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ───────── self‐attention ─────────
class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch//8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch//8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch//2, 1))
        self.v = spectral_norm(nn.Conv2d(ch//2, ch,    1))
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b,c,h,w = x.size()
        f = self.f(x).view(b, -1, h*w)
        g = self.g(x).view(b, -1, h*w)
        beta = torch.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)
        h_ = self.h(x).view(b, -1, h*w)
        o  = torch.bmm(h_, beta).view(b, c//2, h, w)
        return x + self.gamma * self.v(o)

# ───────── mapping ─────────
class Mapping(nn.Sequential):
    def __init__(self, z_dim=128, w_dim=512, n_layers=8):
        layers = []
        for i in range(n_layers):
            inp = z_dim if i==0 else w_dim
            layers += [nn.Linear(inp, w_dim), nn.LeakyReLU(.2)]
        super().__init__(*layers)
        self.apply(he_init)
    def forward(self, z): return pixel_norm(super().forward(z))

# ───────── mod-conv ─────────
class ModConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, demod=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.affine = nn.Linear(512, in_ch)
        self.k, self.pad, self.demod = k, k//2, demod
    def forward(self, x, w):
        b,c,h,w_ = x.shape
        style = self.affine(w).view(b,1,c,1,1) + 1
        wgt = self.weight[None] * style
        if self.demod:
            d = torch.rsqrt((wgt**2).sum([2,3,4]) + 1e-8)
            wgt = wgt * d.view(b,-1,1,1,1)
        wgt = wgt.view(-1,c,self.k,self.k)
        x   = x.view(1,-1,h,w_)
        x   = F.conv2d(x, wgt, padding=self.pad, groups=b)
        return x.view(b,-1,h,w_)

# ───────── G-block ─────────
class GBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = ModConv(in_ch,  out_ch)
        self.c2 = ModConv(out_ch, out_ch)
        self.n1 = nn.Parameter(torch.zeros(1,out_ch,1,1))
        self.n2 = nn.Parameter(torch.zeros(1,out_ch,1,1))
        self.act = nn.LeakyReLU(.2)
    def forward(self, x, w):
        x = self.up(x)
        out1 = self.c1(x, w)
        x = self.act(out1 + self.n1 * torch.randn_like(out1))
        out2 = self.c2(x, w)
        x = self.act(out2 + self.n2 * torch.randn_like(out2))
        return x

# ───────── D-block ─────────
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1   = spectral_norm(nn.Conv2d(in_ch, out_ch, 3,1,1))
        self.c2   = spectral_norm(nn.Conv2d(out_ch,out_ch, 3,1,1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch,1))
        self.pool = nn.AvgPool2d(2)
        self.act  = nn.LeakyReLU(.2)
    def forward(self, x):
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))
class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1   = spectral_norm(nn.Conv2d(in_ch, out_ch, 3,1,1))
        self.c2   = spectral_norm(nn.Conv2d(out_ch,out_ch, 3,1,1))
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch,1))
        self.pool = nn.AvgPool2d(2)
        self.act  = nn.LeakyReLU(.2)
    def forward(self, x):
        y = self.pool(self.act(self.c2(self.act(self.c1(x)))))
        return y + self.pool(self.skip(x))

# ───────── Generator ─────────
class Generator(nn.Module):
    def __init__(self, img_size=(1,128,128), z_dim=128, fmap=128, **_):
        super().__init__()
        self.z_dim   = z_dim
        c,h,_        = img_size
        log_res      = int(math.log2(h))
        self.mapping = Mapping(z_dim)
        self.const   = nn.Parameter(torch.randn(1, fmap*16, 4,4))
        self.upsample= nn.Upsample(scale_factor=2,mode="nearest")

        in_ch = fmap*16
        self.blocks, self.torgb = nn.ModuleList(), nn.ModuleList()
        for i in range(log_res-2):
            out_ch = max(fmap, in_ch//2)
            self.blocks.append(GBlock(in_ch, out_ch))
            if 4*(2**i) in (32,64):
                self.blocks.append(SelfAttention(out_ch))
            self.torgb.append(ModConv(out_ch, c, 1, demod=False))
            in_ch = out_ch
        self.tanh = nn.Tanh()
        self.apply(he_init)

    def forward(self, z):
        w = self.mapping(z)
        x = self.const.expand(z.size(0),-1,-1,-1)
        rgb = None
        for blk, tor in zip(self.blocks, self.torgb):
            x   = blk(x, w)
            rgb = tor(x, w) if rgb is None else self.upsample(rgb) + tor(x, w)
        return self.tanh(rgb)

# ───────── Discriminator ─────────
class Discriminator(nn.Module):
    def __init__(self, img_size=(1,128,128), fmap=128, n_blocks=5, is_critic=False, **_):
        super().__init__()
        c,h,_    = img_size
        log_res  = int(math.log2(h))
        fmap16   = fmap*16

        layers = [spectral_norm(nn.Conv2d(c, fmap, 3,1,1))]
        in_ch  = fmap
        res    = h//2
        res    = h
        for i in range(log_res-2):
            out_ch = min(fmap16, in_ch*2)
            layers.append(DBlock(in_ch, out_ch))
            if res in (64,32):
                layers.append(SelfAttention(out_ch))
            in_ch = out_ch
            res  //= 2

        self.features = nn.Sequential(*layers)
        self.fc       = spectral_norm(nn.Linear(in_ch*res*res, 1))
        self.act_out  = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(he_init)

    def forward(self, x):
        h = self.features(x)
        return self.act_out(self.fc(h.flatten(1))).view(-1)

# ───────── builders ─────────
def build_cxr_g(z_dim=128, base_ch=128):
    return Generator((1,128,128), z_dim=z_dim, fmap=base_ch)

def build_cxr_d(base_ch=128):
    return Discriminator((1,128,128), fmap=base_ch)
