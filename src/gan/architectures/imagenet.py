import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0)
    elif clas1sname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.ReLU(True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)

class Generator(nn.Module):
    def __init__(self, image_size, z_dim=100, filter_dim=64):
        super().__init__()
        # image_size = (3, 224, 224)
        _, H, W = image_size
        # we’ll do 4 up‐sampling steps: 224/(2**4)=14
        init_spatial = H // 16
        dims = [filter_dim, filter_dim*2, filter_dim*4, filter_dim*8, filter_dim*16]
        gen_dims = list(reversed(dims))

        # project z → (gen_dims[0] × init_spatial × init_spatial)
        self.project = nn.Sequential(
            nn.ConvTranspose2d(z_dim, gen_dims[0],
                               kernel_size=init_spatial,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gen_dims[0]),
            nn.ReLU(True),
        )

        # up‐sampling blocks
        blocks = []
        curr_h, curr_w = init_spatial, init_spatial
        for i in range(len(gen_dims)-1):
            in_c, out_c = gen_dims[i], gen_dims[i+1]
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ResidualBlock(in_c),
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True),
            ))
            curr_h *= 2
            curr_w *= 2

        self.gen_blocks = nn.Sequential(*blocks)
        # final conv to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(gen_dims[-1], 3, 3, 1, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, z):
        # z: (B, z_dim)
        x = z.view(-1, z.size(1), 1, 1)
        x = self.project(x)
        x = self.gen_blocks(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, image_size, filter_dim=64, use_batch_norm=True, is_critic=False):
        super().__init__()
        # image_size = (3, 224, 224)
        _, H, W = image_size
        dims = [filter_dim, filter_dim*2, filter_dim*4, filter_dim*8, filter_dim*16]

        blocks = []
        curr_h, curr_w = H, W
        in_c = 3
        for i, out_c in enumerate(dims):
            layers = []
            layers.append(nn.Conv2d(in_c, out_c, 4, 2, 1,
                                    bias=(i==0)))
            if i>0 and use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*layers))
            in_c = out_c
            curr_h //= 2
            curr_w //= 2

        self.conv_blocks = nn.Sequential(*blocks)
        # final “real/fake” conv over the entire spatial map
        self.predict = nn.Conv2d(dims[-1], 1, (curr_h, curr_w), 1, 0, bias=False)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid() if not is_critic else nn.Identity()

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.predict(x)
        x = self.flatten(x).squeeze()
        return self.sigmoid(x)
