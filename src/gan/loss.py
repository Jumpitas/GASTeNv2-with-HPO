import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


def valid_loss(config):
    valid_names = {'wgan-gp', 'ns'}
    if config["name"].lower() not in valid_names:
        return False

    if config["name"].lower() == "wgan-gp":
        return "args" in config and "lambda" in config["args"]

    return True


class DiscriminatorLoss:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        raise NotImplementedError

    def get_loss_terms(self):
        return self.terms


class NS_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self):
        super().__init__([])
        # combine sigmoid + BCE in one go
        self.bce_logits = nn.BCEWithLogitsLoss()

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        ones  = torch.ones_like(real_output, dtype=torch.float, device=device)
        zeros = torch.zeros_like(fake_output, dtype=torch.float, device=device)

        real_loss = self.bce_logits(real_output, ones)
        fake_loss = self.bce_logits(fake_output, zeros)
        return real_loss + fake_loss, {}


class W_DiscrimatorLoss(DiscriminatorLoss):
    # (unused in our current construct_loss — we use WGP instead)
    def __init__(self):
        super().__init__([])

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        d_loss_real = -real_output.mean()
        d_loss_fake =  fake_output.mean()
        return d_loss_real + d_loss_fake, {}


class WGP_DiscriminatorLoss(DiscriminatorLoss):
    def __init__(self, D, lmbda):
        super().__init__(['W_distance', 'D_loss', 'GP'])
        self.D = D
        self.lmbda = lmbda

    def calc_gradient_penalty(self, real_data, fake_data, device):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_data)
        interpolates = (real_data + alpha * (fake_data - real_data)).detach()
        interpolates.requires_grad_(True)

        disc_interpolates = self.D(interpolates)
        grad_outputs = torch.ones_like(disc_interpolates, device=device)

        gradients = autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]

        gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        return ((gradients_norm - 1.)**2).mean()

    def __call__(self, real_data, fake_data, real_output, fake_output, device):
        d_loss_real = -real_output.mean()
        d_loss_fake =  fake_output.mean()
        d_loss = d_loss_real + d_loss_fake

        gp = self.calc_gradient_penalty(real_data, fake_data, device)
        w_distance = - d_loss_real - d_loss_fake

        return d_loss + self.lmbda * gp, {
            'W_distance': w_distance.item(),
            'D_loss':     d_loss.item(),
            'GP':         gp.item()
        }


class GeneratorLoss:
    def __init__(self, terms):
        self.terms = terms

    def __call__(self, device, output):
        raise NotImplementedError

    def get_loss_terms(self):
        return self.terms


class NS_GeneratorLoss(GeneratorLoss):
    def __init__(self):
        super().__init__([])
        self.bce_logits = nn.BCEWithLogitsLoss()

    def __call__(self, device, output):
        ones = torch.ones_like(output, dtype=torch.float, device=device)
        return self.bce_logits(output, ones)


class W_GeneratorLoss(GeneratorLoss):
    def __init__(self):
        super().__init__([])

    def __call__(self, device, output):
        # maximize output.mean()
        return - output.mean()
