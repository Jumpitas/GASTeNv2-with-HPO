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
    HingeR1_DiscriminatorLoss,
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
        # Build for the real dataset size (img_size) and pass z_dim as keyword.
        if isinstance(img_size, int):
            img_shape = (3, img_size, img_size)  # e.g. (3, 32, 32) or (3, 96, 96)
        else:
            img_shape = (3, *img_size)  # already (H, W)
        G = SA_G(
            img_shape,  # first positional = (C,H,W)
            z_dim=config["z_dim"],
            filter_dim=arch["g_filter_dim"],
            n_blocks=arch["g_num_blocks"],
        ).to(device)
        D = SA_D(
            img_shape,
            filter_dim=arch["d_filter_dim"],
            n_blocks=arch["d_num_blocks"],
            is_critic=is_critic,
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
    name = config["name"].lower()
    if name == "ns":
        return NS_GeneratorLoss(), NS_DiscriminatorLoss()

    elif name == "wgan-gp":
        return W_GeneratorLoss(), WGP_DiscriminatorLoss(D, config["args"]["lambda"])

    elif name == "hinge-r1":
        # Hinge discriminator + R1 penalty
        lmbda = config["args"]["lambda"]
        return W_GeneratorLoss(), HingeR1_DiscriminatorLoss(D, lmbda)

    else:
        raise ValueError(f"Unsupported loss: {config['name']}")