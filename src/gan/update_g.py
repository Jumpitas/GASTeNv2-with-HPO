import torch.autograd as autograd
from src.utils.min_norm_solvers import MinNormSolver
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.nn import GaussianNLLLoss, KLDivLoss, BCELoss
from torch import full_like, hstack, log, randn_like


class UpdateGenerator:
    def __init__(self, crit):
        self.crit = crit

    def __call__(self, G, D, optim, noise, device):
        raise NotImplementedError

    def get_loss_terms(self):
        raise NotImplementedError


class UpdateGeneratorGAN(UpdateGenerator):
    def __init__(self, crit):
        super().__init__(crit)

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        loss = self.crit(device, output)
        loss.backward()
        optim.step()
        return loss, {}

    def get_loss_terms(self):
        return []


class UpdateGeneratorGASTEN(UpdateGenerator):
    def __init__(self, crit, C, alpha):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        term_1 = self.crit(device, output)
        # Directly obtain classifier output from the single classifier
        clf_output = self.C(fake_data)
        term_2 = (0.5 - clf_output).abs().mean()
        loss = term_1 + self.alpha * term_2
        loss.backward()
        optim.step()
        return loss, {'original_g_loss': term_1.item(), 'conf_dist_loss': term_2.item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']


class UpdateGeneratorGASTEN_MGDA(UpdateGenerator):
    def __init__(self, crit, C, alpha=1, normalize=False):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.normalize = normalize

    def gradient_normalizers(self, grads, loss):
        return loss.item() * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads]))

    def __call__(self, G, D, optim, noise, device):
        # Compute gradients for term 1 (adversarial loss)
        G.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        term_1 = self.crit(device, output)
        term_1.backward()
        term_1_grads = [param.grad.data.clone() for param in G.parameters() if param.grad is not None]

        # Compute gradients for term 2 (confusion loss using classifier)
        G.zero_grad()
        fake_data = G(noise)
        c_output = self.C(fake_data)
        term_2 = (0.5 - c_output).abs().mean()
        term_2.backward()
        term_2_grads = [param.grad.data.clone() for param in G.parameters() if param.grad is not None]

        if self.normalize:
            gn1 = self.gradient_normalizers(term_1_grads, term_1)
            gn2 = self.gradient_normalizers(term_2_grads, term_2)
            term_1_grads = [gr / gn1 for gr in term_1_grads]
            term_2_grads = [gr / gn2 for gr in term_2_grads]

        scale, min_norm = MinNormSolver.find_min_norm_element([term_1_grads, term_2_grads])

        # Scaled back-propagation
        G.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        term_1 = self.crit(device, output)
        clf_output = self.C(fake_data)
        term_2 = (0.5 - clf_output).abs().mean()
        loss = scale[0] * term_1 + scale[1] * term_2
        loss.backward()
        optim.step()
        return loss, {'original_g_loss': term_1.item(), 'conf_dist_loss': term_2.item(), 'scale1': scale[0],
                      'scale2': scale[1]}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss', 'scale1', 'scale2']


class UpdateGeneratorGASTEN_gaussian(UpdateGenerator):
    def __init__(self, crit, C, alpha, var):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss()
        self.target = 0.5

    def __call__(self, G, D, optim, noise, device):
        # Update based on classifier's Gaussian loss
        G.zero_grad()
        optim.zero_grad()
        fake_data = G(noise)
        clf_output = self.C(fake_data)
        target = full_like(clf_output, fill_value=self.target, device=device)
        var = full_like(clf_output, fill_value=self.var, device=device)
        loss_1 = self.c_loss(clf_output, target, var)
        loss_1.backward()
        clip_grad_norm_(G.parameters(), 0.50 * self.alpha)
        optim.step()
        # Update based on discriminator loss
        optim.zero_grad()
        fake_data = G(noise)
        output = D(fake_data)
        loss_2 = self.crit(device, output)
        loss_2.backward()
        clip_grad_norm_(G.parameters(), 0.50)
        optim.step()
        loss = loss_1 + loss_2
        return loss, {'original_g_loss': loss_2.item(), 'conf_dist_loss': loss_1.item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']


class UpdateGeneratorGASTEN_KLDiv(UpdateGenerator):
    def __init__(self, crit, C, alpha):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.c_loss = KLDivLoss(reduction="none")
        self.target = 0.5
        self.eps = 1e-9
        self.crit = BCELoss(reduction="none")

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        optim.zero_grad()
        fake_data = G(noise)
        # For a single classifier, obtain the output directly
        clf_output = self.C(fake_data)
        target = full_like(clf_output, fill_value=self.target, device=device)
        loss_1 = self.c_loss(log(clf_output.clip(min=self.eps, max=1.0)), target).mean()
        output = D(fake_data)
        target_out = full_like(output, fill_value=1.0, device=device)
        loss_2 = self.crit(output, target_out).mean()
        loss = (self.alpha * loss_1) + loss_2
        loss.backward()
        optim.step()
        return loss, {'original_g_loss': loss_2.item(), 'conf_dist_loss': loss_1.item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']


class UpdateGeneratorGASTEN_gaussianV2(UpdateGenerator):
    def __init__(self, crit, C, alpha, var):
        super().__init__(crit)
        self.C = C
        self.alpha = alpha
        self.var = var
        self.c_loss = GaussianNLLLoss(reduction="none")
        self.target = 0.5
        self.crit = BCELoss(reduction="none")

    def __call__(self, G, D, optim, noise, device):
        G.zero_grad()
        optim.zero_grad()
        fake_data = G(noise)
        # For a single classifier, assume the classifier returns a tuple where index 0 is the prediction
        clf_output = self.C(fake_data, output_feature_maps=True)
        # Directly apply GaussianNLLLoss on the prediction tensor
        loss_1 = self.c_loss(clf_output[0], full_like(clf_output[0], fill_value=self.target, device=device),
                             full_like(clf_output[0], fill_value=self.var, device=device)).mean()
        output = D(fake_data)
        target = full_like(output, fill_value=1.0, device=device)
        loss_2 = self.crit(output, target).mean()
        loss = (self.alpha * loss_1) + loss_2
        loss.backward()
        optim.step()
        return loss, {'original_g_loss': loss_2.item(), 'conf_dist_loss': loss_1.item()}

    def get_loss_terms(self):
        return ['original_g_loss', 'conf_dist_loss']
