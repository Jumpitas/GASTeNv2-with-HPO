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

        # try feature‐map API, otherwise plain forward
        try:
            out = self.C(images, output_feature_maps=True)[0]
        except TypeError:
            out = self.C(images)

        # if it's a [B, 2] logits tensor, turn into positive‐class prob
        if out.ndim == 2 and out.size(1) == 2:
            preds = F.softmax(out, dim=1)[:, 1]
        else:
            # otherwise assume feature‐maps: pool down to a scalar per sample
            # now out.ndim >= 3
            pooled = F.adaptive_avg_pool2d(out, (1, 1))  # → [B, C,1,1]
            pooled = pooled.flatten(1).mean(dim=1)       # → [B]
            preds = pooled

        # store on CPU
        self.preds[start_idx:start_idx + batch_size] = preds.cpu()

    def _kl(self, p, q):
        return kl_divergence(Categorical(probs=p),
                             Categorical(probs=q)).mean()

    def compute(self, ref_preds=None):
        p = self.preds
        binary = torch.stack((p, 1.0 - p), dim=1)

        if ref_preds is None:
            ref = torch.full_like(binary, 0.5)
        else:
            ref = torch.stack((ref_preds, 1.0 - ref_preds), dim=1)

        worst = torch.linspace(0.0, 1.0, p.size(0), device=p.device)
        worst = torch.stack((worst, 1.0 - worst), dim=1)

        ref_kl = self._kl(worst, ref)
        amb_kl = self._kl(binary, ref)

        return 1.0 - torch.exp(-(amb_kl / ref_kl))

    def finalize(self):
        val = self.compute()
        self.result = val.item() if isinstance(val, torch.Tensor) else float(val)
        return self.result

    def get_clfs(self):
        return []

    def reset(self):
        self.result = 1.0
        self.preds = torch.zeros(self.dataset_size, dtype=torch.float32)
