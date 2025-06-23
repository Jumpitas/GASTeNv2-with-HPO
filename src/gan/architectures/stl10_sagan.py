import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# -----------------------------------------------------------------------------
# Weight init (as in original SAGAN)
# -----------------------------------------------------------------------------
def weights_init(m):
    cname = m.__class__.__name__
    if cname.find('Conv') != -1 or cname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    elif cname.find('BatchNorm') != -1 or cname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

# -----------------------------------------------------------------------------
# Self‐Attention block (SAGAN)
# -----------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch//8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch//8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch//2, 1))
        self.v = spectral_norm(nn.Conv2d(ch//2, ch,    1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        f = self.f(x).view(B, -1, H*W)                   # B×C/8×HW
        g = self.g(x).view(B, -1, H*W)                   # B×C/8×HW
        beta = torch.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)  # B×HW×HW
        h_ = self.h(x).view(B, -1, H*W)                  # B×C/2×HW
        o  = torch.bmm(h_, beta).view(B, C//2, H, W)     # B×C/2×H×W
        o  = self.v(o)                                   # B×C×H×W
        return x + self.gamma * o

# -----------------------------------------------------------------------------
# Residual Up / Down blocks
# -----------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        if use_sn:
            conv = spectral_norm(conv)
        self.net = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------------------------------------------------------
# Generator (unchanged from original stl10_sagan)
# -----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(
        self,
        image_size: tuple[int,int,int]=(3,96,96),
        z_dim: int=100,
        filter_dim: int=64,
        n_blocks: int=5
    ):
        super().__init__()
        C,H,W = image_size
        max_blocks = max(int(math.log2(min(H,W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)

        # compute initial feature‐map size
        self.init_h = H // (2**self.n_blocks)
        self.init_w = W // (2**self.n_blocks)
        self.init_ch= filter_dim * (2**self.n_blocks)

        self.z_dim = z_dim
        self.project = nn.Sequential(
            nn.Linear(z_dim, self.init_ch*self.init_h*self.init_w, bias=False),
            nn.BatchNorm1d(self.init_ch*self.init_h*self.init_w),
            nn.ReLU(True),
        )

        blocks = []
        ch = self.init_ch
        for i in range(self.n_blocks):
            nxt = ch//2
            blocks.append(UpBlock(ch, nxt))
            ch = nxt
            # insert attention at 32×32 and 64×64
            if self.init_h * (2**(i+1)) in (32,64):
                blocks.append(SelfAttention(ch))
        self.up_blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, C, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, self.init_ch, self.init_h, self.init_w)
        x = self.up_blocks(x)
        return self.to_rgb(x)

# -----------------------------------------------------------------------------
# Discriminator with dynamic final‐conv kernel
# -----------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(
        self,
        image_size: tuple[int,int,int]=(3,96,96),
        filter_dim: int=64,
        n_blocks: int=5,
        use_sn: bool=True,
        is_critic: bool=False
    ):
        super().__init__()
        C,H,W = image_size
        max_blocks = max(int(math.log2(min(H,W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)

        # build downsampling path
        layers = []
        in_ch = C
        for i in range(self.n_blocks):
            out_ch = filter_dim * (2**i)
            layers.append(DownBlock(in_ch, out_ch, use_sn))
            in_ch = out_ch
            # attention at 32×32 and 64×64
            res = H // (2**(i+1))
            if res in (64,32):
                layers.append(SelfAttention(in_ch))
        self.features = nn.Sequential(*layers)

        # final feature‐map size
        final_res = H // (2**self.n_blocks)
        # global head: conv with kernel = final_res
        conv = nn.Conv2d(in_ch, 1, kernel_size=final_res, stride=1, padding=0, bias=False)
        if use_sn:
            conv = spectral_norm(conv)
        self.global_head = conv

        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)                  # B×in_ch×final_res×final_res
        out = self.global_head(h)             # B×1×1×1
        out = out.view(x.size(0), -1)         # B×1
        return self.act_out(out).view(-1)     # B

