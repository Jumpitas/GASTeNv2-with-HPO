import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from .metric import Metric

class Hubris(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.reset()

    @torch.no_grad()
    def update(self, images, batch):
        start_idx, batch_size = batch
        output = self.C(images, output_feature_maps=True)[0]

        if output.dim() > 1:
            output = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1).mean(dim=1)

        self.preds[start_idx:start_idx + batch_size] = output.squeeze().cpu()

    def _kl(self, p, q):
        return kl_divergence(Categorical(probs=p), Categorical(probs=q)).mean()

    def compute(self, ref_preds=None):
        """Compute hubris from the stored self.preds vector."""
        preds = self.preds
        binary_preds = torch.stack((preds, 1.0 - preds), dim=1)

        if ref_preds is None:
            reference = torch.full_like(binary_preds, 0.5)
        else:
            reference = torch.stack((ref_preds, 1.0 - ref_preds), dim=1)

        worst = torch.linspace(0.0, 1.0, preds.size(0), device=preds.device)
        worst = torch.stack((worst, 1.0 - worst), dim=1)

        ref_kl = self._kl(worst, reference)
        amb_kl = self._kl(binary_preds, reference)

        return 1.0 - torch.exp(-(amb_kl / ref_kl))

    def finalize(self):
        # no args, compute uses self.preds
        val = self.compute()
        # unwrap tensor → Python float
        self.result = val.item() if isinstance(val, torch.Tensor) else float(val)
        return self.result

    def get_clfs(self):
        return []

    def reset(self):
        self.result = 1.0
        self.preds = torch.zeros(self.dataset_size, dtype=torch.float32)
