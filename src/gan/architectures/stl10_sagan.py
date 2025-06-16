import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator96(nn.Module):
    def __init__(self, z_dim=100, base_ch=64, img_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            # z → (batch, base_ch*8, 6, 6)
            nn.ConvTranspose2d(z_dim, base_ch * 8, kernel_size=6, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(True),

            # →12×12
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),

            # →24×24
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),

            # →48×48
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            # →96×96
            nn.ConvTranspose2d(base_ch, base_ch//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch//2),
            nn.ReLU(True),

            # final conv to RGB
            nn.Conv2d(base_ch//2, img_ch, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # z: (batch, z_dim)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator96(nn.Module):
    def __init__(self, base_ch=64, img_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            # 96×96
            nn.Conv2d(img_ch, base_ch//2, 4, 2, 1, bias=False),   # →48×48
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch//2, base_ch, 4, 2, 1, bias=False),  # →24×24
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1, bias=False),   # →12×12
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1, bias=False), # →6×6
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),

            # final “validity” conv
            nn.Conv2d(base_ch*4, 1, 6, 1, 0, bias=False),        # →1×1
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)


if __name__ == "__main__":
    # sanity‐check memory on a single 96×96 forward pass
    G = Generator96().cuda()
    D = Discriminator96().cuda()

    # check inference in moderate batches
    with torch.no_grad():
        for _ in range((10000 // 128) + 1):
            z = torch.randn(128, 100, device="cuda")
            fake = G(z)
            real_pred = D(fake)
