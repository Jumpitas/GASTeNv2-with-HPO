import torch
import matplotlib.pyplot as plt
import seaborn as sns
from .metric import Metric
from .hubris import Hubris
from torchvision.transforms import ToTensor
import PIL
import pandas as pd


class OutputsHistogram(Metric):
    def __init__(self, C, dataset_size):
        super().__init__()
        self.C = C
        self.dataset_size = dataset_size
        self.y_hat = torch.zeros(dataset_size, dtype=torch.float)
        self.to_tensor = ToTensor()
        self.hubris = Hubris(C, dataset_size)
        self.reset()

    def update(self, images, batch):
        start_idx, batch_size = batch
        # Update hubris with the images (single classifier mode)
        self.hubris.update(images, batch)
        with torch.no_grad():
            # Call the classifier directly with output_feature_maps=True.
            # Assume it returns a tuple: (prediction, feature_map)
            output = self.C(images, output_feature_maps=True)
            c_output = output[0]
            # If output is not 1D per sample, pool spatial dimensions:
            if c_output.dim() > 1:
                # Apply adaptive average pooling to reduce spatial dims to (1,1)
                c_output = torch.nn.functional.adaptive_avg_pool2d(c_output, (1, 1))
                # Flatten to shape [batch, channels]
                c_output = c_output.view(c_output.size(0), -1)
                # If multiple channels, average over them to get a scalar per image
                if c_output.size(1) > 1:
                    c_output = c_output.mean(dim=1)
            # Now, c_output should be of shape [batch]
        self.y_hat[start_idx:start_idx + batch_size] = c_output

    def plot(self):
        # Plot a histogram of the classifier's outputs
        plt.figure()
        sns.histplot(self.y_hat.cpu().numpy(), stat='proportion', bins=20)
        plt.title("Classifier Output Distribution")
        plt.xlabel("Prediction")
        plt.ylabel("Proportion")
        plt.show()

    def plot_clfs(self):
        # In single-classifier mode, plot:
        # (1) the output distribution,
        # (2) the confusion distance (|0.5 - prediction|),
        # (3) a bar for the computed hubris value.
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        sns.kdeplot(self.y_hat.cpu().numpy(), ax=axs[0], fill=True)
        axs[0].set(xlim=(0.0, 1.0), title="Classifier Output Distribution")

        cd = torch.abs(0.5 - self.y_hat)
        sns.kdeplot(cd.cpu().numpy(), ax=axs[1], fill=True)
        axs[1].set(xlim=(0.0, 0.5), title="Confusion Distance Distribution")

        hubris_value = self.hubris.finalize()
        axs[2].bar(['Hubris'], [hubris_value])
        axs[2].set(ylim=(0, 1), title="Hubris Value")

        fig.canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        return self.to_tensor(pil_image)

    def reset(self):
        self.y_hat = torch.zeros(self.dataset_size, dtype=torch.float)
        self.hubris.reset()
