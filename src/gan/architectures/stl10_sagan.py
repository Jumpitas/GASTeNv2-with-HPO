import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# --------------------------------------------------
# Self-Attention block (from SAGAN)
# --------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.g = spectral_norm(nn.Conv2d(ch,   ch // 8, 1))
        self.h = spectral_norm(nn.Conv2d(ch,   ch // 2, 1))
        self.v = spectral_norm(nn.Conv2d(ch // 2, ch,   1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        f = self.f(x).view(B, -1, H*W)            # B×(C/8)×(HW)
        g = self.g(x).view(B, -1, H*W)            # B×(C/8)×(HW)
        beta = torch.softmax(torch.bmm(f.permute(0,2,1), g), dim=-1)  # B×(HW)×(HW)
        h_ = self.h(x).view(B, -1, H*W)           # B×(C/2)×(HW)
        o = torch.bmm(h_, beta).view(B, C//2, H, W)
        o = self.v(o)
        return x + self.gamma * o

# --------------------------------------------------
# Residual Up / Down blocks
# --------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = spectral_norm(nn.Conv2d(in_ch,  out_ch, 3,1,1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3,1,1))
        self.skip  = spectral_norm(nn.Conv2d(in_ch,  out_ch, 1,1,0))
        self.norm1 = nn.InstanceNorm2d(in_ch, affine=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.upsample(x)))
        y = self.conv1(y)
        y = self.act(self.norm2(y))
        y = self.conv2(y)
        skip = self.skip(self.upsample(x))
        return y + skip

class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_sn: bool = True):
        super().__init__()
        Conv = (lambda *a,**k: spectral_norm(nn.Conv2d(*a,**k))) if use_sn else nn.Conv2d
        self.conv1 = Conv(in_ch, out_ch, 3,1,1)
        self.conv2 = Conv(out_ch, out_ch, 3,1,1)
        self.skip  = Conv(in_ch,  out_ch, 1,1,0)
        self.pool  = nn.AvgPool2d(2)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.pool(y)
        skip = self.pool(self.skip(x))
        return y + skip

# --------------------------------------------------
# Generator
# --------------------------------------------------
class Generator(nn.Module):
    def __init__(
        self,
        img_ch: int = 3,         # ← now accepts img_ch
        z_dim: int = 128,
        filter_dim: int = 64,
        n_blocks: int = 4,
        **_
    ):
        super().__init__()
        # STL-10 is 96×96  ⇒  n_blocks=4 gives 96 = 6*(2**4)
        C, H, W = img_ch, 96, 96

        # clamp blocks so final feature map ≥ 4×4
        max_blocks = max(int(math.log2(min(H, W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)

        self.init_h  = H // (2**self.n_blocks)
        self.init_w  = W // (2**self.n_blocks)
        self.init_ch = filter_dim * (2**self.n_blocks)

        # map z → initial feature map
        self.project = nn.Sequential(
            nn.Linear(z_dim, self.init_ch * self.init_h * self.init_w),
            nn.ReLU(True)
        )

        # build upsampling stacks (with attention at 32 & 64 px)
        blocks = []
        ch = self.init_ch
        for i in range(self.n_blocks):
            nxt = ch // 2
            blocks.append(UpBlock(ch, nxt))
            ch = nxt
            if self.init_h * (2**(i+1)) in (32, 64):
                blocks.append(SelfAttention(ch))
        self.up_blocks = nn.Sequential(*blocks)

        # final RGB conv
        self.to_rgb = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, img_ch, 3,1,1)),
            nn.Tanh()
        )

        self.z_dim = z_dim
        self.apply(self._weights_init)

    def _weights_init(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        x = self.project(z).view(b, self.init_ch, self.init_h, self.init_w)
        x = self.up_blocks(x)
        return self.to_rgb(x)

# --------------------------------------------------
# Discriminator
# --------------------------------------------------
class Discriminator(nn.Module):
    def __init__(
        self,
        img_ch: int = 3,         # ← now accepts img_ch
        filter_dim: int = 64,
        n_blocks: int = 4,
        is_critic: bool = False,
        **_
    ):
        super().__init__()
        # again assume 96×96
        C, H, W = img_ch, 96, 96

        max_blocks = max(int(math.log2(min(H,W))) - 2, 1)
        self.n_blocks = min(n_blocks, max_blocks)

        # initial downsample
        ch = filter_dim
        blocks = [DownBlock(C, ch)]
        res = H // 2

        # add more blocks + self-attention at 64,32
        for i in range(1, self.n_blocks):
            nxt = ch * 2
            blocks.append(DownBlock(ch, nxt))
            ch = nxt
            res //= 2
            if res in (64, 32):
                blocks.append(SelfAttention(ch))
        self.features = nn.Sequential(*blocks)

        # global head
        self.global_head = spectral_norm(nn.Conv2d(ch, 1, res, 1, 0))
        # 70×70 patch head
        self.patch_head  = spectral_norm(nn.Conv2d(ch, 1, 4,   1, 0))
        self.act_out = nn.Identity() if is_critic else nn.Sigmoid()

        self.apply(self._weights_init)

    def _weights_init(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.features(x)
        glob = self.global_head(h).view(x.size(0), -1)
        patch= self.patch_head(h).mean([2,3])
        out  = glob + patch
        return self.act_out(out).view(-1)