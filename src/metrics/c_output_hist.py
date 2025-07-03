import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import PIL
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
        # update hubris metric
        self.hubris.update(images, batch)

        # get classifier outputs
        logits = self.C(images)
        if logits.dim() > 1 and logits.size(1) > 1:
            probs = F.softmax(logits, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(logits).view(-1)

        # store in the running buffer
        self.y_hat[start:start + bs] = probs.cpu()

    def _var_safe_plot(self, data, ax, title, bins=20, xlim=None):
        # choose histogram vs KDE depending on variance
        if np.var(data) < 1e-6:
            ax.hist(data, bins=bins, density=True)
        else:
            import seaborn as sns
            sns.kdeplot(data, ax=ax, fill=True, warn_singular=False)
        ax.set(title=title)
        if xlim:
            ax.set_xlim(*xlim)

    def plot(self):
        # simple single‐plot view
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
        # 3‐panel figure: output dist, confusion dist, hubris
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        self._var_safe_plot(
            self.y_hat.numpy(),
            axs[0],
            "Classifier Output Distribution",
            bins=20,
            xlim=(0, 1),
        )


        cd = np.abs(0.5 - self.y_hat.numpy())
        self._var_safe_plot(
            cd,
            axs[1],
            "Confusion Distance Distribution",
            bins=20,
            xlim=(0, 0.5),
        )

        hubris_val = float(self.hubris.finalize())
        axs[2].bar(["Hubris"], [hubris_val])
        axs[2].set_ylim(0, 1)

        fig.canvas.draw()

        buf, (w, h) = fig.canvas.print_to_buffer()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))

        img = PIL.Image.fromarray(arr, mode="RGBA").convert("RGB")
        plt.close(fig)
        return self.to_tensor(img)

    def reset(self):
        self.y_hat.zero_()
        self.hubris.reset()
