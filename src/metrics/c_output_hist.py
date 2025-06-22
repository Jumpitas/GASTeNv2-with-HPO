import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

        logits = self.C(images)
        if logits.dim() > 1 and logits.size(1) > 1:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits).view(-1)
        self.y_hat[start:start + bs] = probs.cpu()

    def _var_safe_plot(self, data, ax, title, bins=20, xlim=None):
        if np.var(data) < 1e-6:
            ax.hist(data, bins=bins, density=True)
        else:
            import seaborn as sns
            sns.kdeplot(data, ax=ax, fill=True, warn_singular=False)
        ax.set(title=title)
        if xlim:
            ax.set_xlim(*xlim)

    def plot(self):
        plt.figure()
        self._var_safe_plot(
            self.y_hat.numpy(),
            plt.gca(),
            "Classifier Output Distribution",
            bins=20,
            xlim=(0, 1),
        )
        plt.xlabel("Prediction")
        plt.ylabel("Proportion")
        plt.show()

    def plot_clfs(self):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # 1) output distribution
        self._var_safe_plot(
            self.y_hat.numpy(),
            axs[0],
            "Classifier Output Distribution",
            bins=20,
            xlim=(0, 1),
        )

        # 2) confusion distance
        cd = np.abs(0.5 - self.y_hat.numpy())
        self._var_safe_plot(
            cd,
            axs[1],
            "Confusion Distance Distribution",
            bins=20,
            xlim=(0, .5),
        )

        # 3) hubris bar
        hubris_val = float(self.hubris.finalize())
        axs[2].bar(['Hubris'], [hubris_val])
        axs[2].set_ylim(0, 1)

        # draw & grab RGBA buffer
        fig.canvas.draw()
        buf = fig.canvas.renderer.buffer_rgba()   # memoryview
        arr = np.asarray(buf)                     # now a numpy array
        img = PIL.Image.fromarray(arr, mode='RGBA').convert('RGB')
        plt.close(fig)

        return self.to_tensor(img)

    def reset(self):
        self.y_hat.zero_()
        self.hubris.reset()
