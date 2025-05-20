import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL
from torchvision.transforms import ToTensor

from .metric import Metric
from .hubris import Hubris


class OutputsHistogram(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.y_hat = torch.zeros(dataset_size, dtype=torch.float32)
        self.to_tensor = ToTensor()
        self.hubris = Hubris(C, dataset_size)

    @torch.no_grad()
    def update(self, images, batch):
        start, bs = batch
        self.hubris.update(images, batch)

        out = self.C(images, output_feature_maps=True)[0]
        if out.dim() > 1:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1).mean(dim=1)
        self.y_hat[start:start + bs] = out.squeeze().cpu()

    def _var_safe_plot(self, data, ax, title, bins=20, xlim=None):
        if np.var(data) < 1e-6:
            sns.histplot(data, ax=ax, stat='proportion', bins=bins)
        else:
            sns.kdeplot(data, ax=ax, fill=True, warn_singular=False)
        ax.set(title=title)
        if xlim:
            ax.set_xlim(*xlim)

    def plot(self):
        plt.figure()
        self._var_safe_plot(self.y_hat.numpy(), plt.gca(),
                            "Classifier Output Distribution", bins=20, xlim=(0, 1))
        plt.xlabel("Prediction"); plt.ylabel("Proportion"); plt.show()

    def plot_clfs(self):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        self._var_safe_plot(self.y_hat.numpy(), axs[0],
                            "Classifier Output Distribution", bins=20, xlim=(0, 1))

        cd = torch.abs(0.5 - self.y_hat).numpy()
        self._var_safe_plot(cd, axs[1],
                            "Confusion Distance Distribution", bins=20, xlim=(0, .5))

        hubris_val = self.hubris.finalize().item()
        axs[2].bar(['Hubris'], [hubris_val]); axs[2].set_ylim(0, 1)

        fig.canvas.draw()
        img = PIL.Image.frombytes('RGB',
                                  fig.canvas.get_width_height(),
                                  fig.canvas.tostring_rgb())
        plt.close(fig)
        return self.to_tensor(img)

    def reset(self):
        self.y_hat.zero_()
        self.hubris.reset()
