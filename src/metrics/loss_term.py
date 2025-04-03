import torch
from .metric import Metric

class LossSecondTerm(Metric):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.count = 0
        self.acc = 0
        self.result = float('inf')

    def update(self, images, batch):
        start_idx, batch_size = batch
        with torch.no_grad():
            # Call the classifier with output_feature_maps=True.
            # Instead of unpacking, we take the first element as the prediction.
            output = self.C(images, output_feature_maps=True)
            c_output = output[0]
        term_2 = (0.5 - c_output).sum().abs().item()
        self.acc += term_2
        self.count += images.size(0)

    def finalize(self):
        self.result = self.acc / self.count
        return self.result

    def reset(self):
        self.count = 0
        self.acc = 0
        self.result = float('inf')
