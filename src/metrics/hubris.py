from .metric import Metric
import torch
import torch.nn.functional as F

class Hubris(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        # In single-classifier mode, there is no ensemble, so we set output_clfs to 0.
        self.output_clfs = 0
        self.reset()

    def update(self, images, batch):
        start_idx, batch_size = batch
        with torch.no_grad():
            # Directly call the classifier with output_feature_maps=True.
            # Assume that calling self.C(images, output_feature_maps=True) returns a tuple,
            # where the first element is the prediction tensor.
            output = self.C(images, output_feature_maps=True)
            # Use the first element as the prediction.
            c_output = output[0]
            # If c_output has more than one dimension, pool over the spatial dimensions and average over channels
            if c_output.dim() > 1:
                # Adaptive average pool to reduce spatial dims to (1, 1)
                c_output = F.adaptive_avg_pool2d(c_output, (1, 1))
                # Flatten to shape [batch, channels]
                c_output = c_output.view(c_output.size(0), -1)
                # If multiple channels, average over them to get one scalar per image
                c_output = c_output.mean(dim=1)
            # Ensure c_output is 1D (shape: [batch])
            c_output = c_output.squeeze()
        # Store the classifier output in preds
        self.preds[start_idx:start_idx + batch_size] = c_output

    def compute(self, preds, ref_preds=None):
        # Compute a worst-case prediction linearly spanning [0,1]
        worst_preds = torch.linspace(0.0, 1.0, steps=preds.size(0))
        binary_preds = torch.hstack((preds, 1.0 - preds))
        if ref_preds is not None:
            reference = ref_preds.clone()
            reference = torch.hstack((reference, 1.0 - reference))
        else:
            reference = torch.full_like(binary_preds, fill_value=0.50)
        binary_worst_preds = torch.hstack((worst_preds, 1.0 - worst_preds))
        predictions_full = torch.distributions.Categorical(probs=binary_preds)
        reference_full = torch.distributions.Categorical(probs=reference)
        worst_preds_full = torch.distributions.Categorical(probs=binary_worst_preds)
        ref_kl = torch.distributions.kl.kl_divergence(worst_preds_full, reference_full).mean()
        amb_kl = torch.distributions.kl.kl_divergence(predictions_full, reference_full).mean()
        return 1.0 - torch.exp(-(amb_kl / ref_kl)).item()

    def finalize(self):
        self.result = self.compute(self.preds)
        return self.result

    def get_clfs(self):
        # In single classifier mode, no separate classifier outputs exist.
        return []

    def reset(self):
        self.result = torch.tensor([1.0])
        self.preds = torch.zeros(self.dataset_size, dtype=torch.float)
