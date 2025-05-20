# src/gan/__init__.py

from src.gan.architectures.dcgan import Generator as DC_G, Discriminator as DC_D
from src.gan.architectures.dcgan_v2 import Generator as DC_G2, Discriminator as DC_D2
from src.gan.architectures.resnet import Generator as RN_G, Discriminator as RN_D
from src.gan.architectures.chest_xray import Generator as CXR_G, Discriminator as CXR_D
from src.gan.architectures.stl10_sagan import Generator as SA_G, Discriminator as SA_D
from src.gan.architectures.imagenet import Generator as Imagenet_G, Discriminator as Imagenet_D

from src.gan.loss import (
    NS_GeneratorLoss,
    NS_DiscriminatorLoss,
    W_GeneratorLoss,
    WGP_DiscriminatorLoss,
)


def construct_gan(config, img_size, device):
    use_batch_norm = config["loss"]["name"] != "wgan-gp"
    is_critic      = config["loss"]["name"] == "wgan-gp"
    arch = config["architecture"]

    if arch["name"] == "dcgan":
        G = DC_G(
            img_size,
            z_dim=config["z_dim"],
            filter_dim=arch["g_filter_dim"],
            n_blocks=arch["g_num_blocks"],
        ).to(device)
        D = DC_D(
            img_size,
            filter_dim=arch["d_filter_dim"],
            n_blocks=arch["d_num_blocks"],
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "dcgan-v2":
        G = DC_G2(
            img_size,
            z_dim=config["z_dim"],
            filter_dim=arch["g_filter_dim"],
            n_blocks=arch["g_num_blocks"],
        ).to(device)
        D = DC_D2(
            img_size,
            filter_dim=arch["d_filter_dim"],
            n_blocks=arch["d_num_blocks"],
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "resnet":
        G = RN_G(
            img_size,
            z_dim=config["z_dim"],
            gf_dim=arch["g_filter_dim"],
        ).to(device)
        D = RN_D(
            img_size,
            df_dim=arch["d_filter_dim"],
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "chest-xray":
        G = CXR_G(
            img_size,
            z_dim=config["z_dim"],
            filter_dim=arch["g_filter_dim"],
            n_blocks=arch["g_num_blocks"],
        ).to(device)
        D = CXR_D(
            img_size,
            filter_dim=arch["d_filter_dim"],
            n_blocks=arch["d_num_blocks"],
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(device)

    elif arch["name"] == "stl10_sagan":
        G = SA_G(
            config["z_dim"],
            arch["g_filter_dim"],
            img_ch=3
        ).to(device)
        G.z_dim = config["z_dim"]
        D = SA_D(
            arch["d_filter_dim"],
            img_ch=3
        ).to(device)

    elif arch["name"] == "imagenet":
        G = Imagenet_G(
            img_size,
            z_dim=config["z_dim"],
            filter_dim=arch["g_filter_dim"],
        ).to(device)
        D = Imagenet_D(
            img_size,
            filter_dim=arch["d_filter_dim"],
            use_batch_norm=use_batch_norm,
            is_critic=is_critic,
        ).to(device)

    else:
        raise ValueError(f"Unsupported architecture: {arch['name']}")

    return G, D


def construct_loss(config, D):
    if config["name"] == "ns":
        return NS_GeneratorLoss(), NS_DiscriminatorLoss()
    elif config["name"] == "wgan-gp":
        return W_GeneratorLoss(), WGP_DiscriminatorLoss(D, config["args"]["lambda"])
    else:
        raise ValueError(f"Unsupported loss: {config['name']}")
