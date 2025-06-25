from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ───────────────────────── helpers ───────────────────────────
def he(m: nn.Module) -> None:        # Kaiming-He init
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ───────────────────── Self-Attention ────────────────────────
class SelfAttention(nn.Module):
    """SAGAN attention block (no spatial change)."""
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch, ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch, ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch, ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        f = self.f(x).view(b, -1, h * w)             # B × C/8 × HW
        g = self.g(x).view(b, -1, h * w)             # B × C/8 × HW
        beta = torch.softmax(torch.bmm(f.permute(0, 2, 1), g), -1)  # B × HW × HW
        h_ = self.h(x).view(b, -1, h * w)            # B × C/2 × HW
        o = torch.bmm(h_, beta).view(b, c // 2, h, w)  # B × C/2 × H × W
        return x + self.gamma * self.v(o)


# ───────────────────── Residual blocks ───────────────────────
class UpBlock(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.c1 = nn.Conv2d(ic, oc, 3, 1, 1)
        self.c2 = nn.Conv2d(oc, oc, 3, 1, 1)
        self.skip = nn.Conv2d(ic, oc, 1)
        self.bn1 = nn.InstanceNorm2d(ic, affine=True)
        self.bn2 = nn.InstanceNorm2d(oc, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.up(x)))
        y = self.c1(y)
        y = self.act(self.bn2(y))
        y = self.c2(y)
        return y + self.skip(self.up(x))


class DownBlock(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv2d(ic, oc, 3, 1, 1))
        self.c2 = spectral_norm(nn.Conv2d(oc, oc, 3, 1, 1))
        self.skip = spectral_norm(nn.Conv2d(ic, oc, 1))
        self.pool = nn.AvgPool2d(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.c1(x))
        y = self.act(self.c2(y))
        y = self.pool(y)
        return y + self.pool(self.skip(x))


# ───────────────────────── Generator ─────────────────────────
class Generator(nn.Module):
    """
    SAGAN-style generator.  Z-dim is stored for external access (HPO code).
    """
    def __init__(self,
                 img_size: tuple[int, int, int] | int = (3, 96, 96),
                 z_dim: int = 128,
                 filter_dim: int = 64,
                 n_blocks: int = 4,
                 **_):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (3, img_size, img_size)
        c, h, _ = img_size
        self.z_dim = z_dim

        init_h = h // (2 ** n_blocks)
        init_ch = filter_dim * (2 ** n_blocks)

        self.project = nn.Sequential(
            nn.Linear(z_dim, init_ch * init_h * init_h),
            nn.LeakyReLU(0.2, inplace=True)
        )

        blocks, ch = [], init_ch
        for i in range(n_blocks):
            nxt = ch // 2
            blocks.append(UpBlock(ch, nxt))
            ch = nxt
            if init_h * (2 ** (i + 1)) in {32, 64}:
                blocks.append(SelfAttention(ch))
        self.ups = nn.Sequential(*blocks)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, c, 3, 1, 1),
            nn.Tanh()
        )
        self.apply(he)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        x = self.project(z).view(b, -1, 6, 6)  # 6×6 is init_h for 96×96 @ n_blocks=4
        return self.to_rgb(self.ups(x))


# ──────────────────────── Discriminator ─────────────────────
class Discriminator(nn.Module):
    """
    Residual discriminator with SA blocks and correct final in_features.
    """
    def __init__(self,
                 img_size: tuple[int, int, int] | int = (3, 96, 96),
                 filter_dim: int = 64,
                 n_blocks: int = 4,
                 is_critic: bool = False,
                 **_):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (3, img_size, img_size)
        c, h, _ = img_size

        layers = [DownBlock(c, filter_dim)]
        res = h // 2                      # after first DownBlock
        ch = filter_dim

        for _ in range(1, n_blocks):
            nxt = ch * 2
            layers.append(DownBlock(ch, nxt))
            ch = nxt
            res //= 2
            if res in {64, 32}:
                layers.append(SelfAttention(ch))

        self.feats = nn.Sequential(*layers)
        self.final = spectral_norm(nn.Linear(ch * res * res, 1))
        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(he)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feats(x)
        return self.act_out(self.final(h.flatten(1))).view(-1)


# ───────────────────── convenience builders ─────────────────
def build_stl10_g(z_dim: int = 128, base_ch: int = 64) -> Generator:
    return Generator((3, 96, 96), z_dim=z_dim, filter_dim=base_ch, n_blocks=4)


def build_stl10_d(base_ch: int = 64, critic: bool = False) -> Discriminator:
    return Discriminator((3, 96, 96), filter_dim=base_ch,
                         n_blocks=4, is_critic=critic)