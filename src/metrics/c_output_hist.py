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
        # Initialize a 1D tensor to store predictions for each sample.
        self.y_hat = torch.zeros(dataset_size, dtype=torch.float)
        self.to_tensor = ToTensor()
        # Create an instance of the hubris metric.
        self.hubris = Hubris(C, dataset_size)
        self.reset()

    def update(self, images, batch):
        start_idx, batch_size = batch
        # Update the hubris metric using the current batch.
        self.hubris.update(images, batch)
        with torch.no_grad():
            # Call the classifier with output_feature_maps=True.
            # The expected behavior is to return a tuple: (prediction, feature_map).
            output = self.C(images, output_feature_maps=True)
            # If the output is a tuple or list, take the first element as predictions.
            if isinstance(output, (tuple, list)):
                c_output = output[0]
            else:
                c_output = output
            # If the prediction has more than one dimension (e.g. [batch, channels, h, w]),
            # use adaptive average pooling to reduce spatial dimensions and average over channels.
            if c_output.dim() > 1:
                c_output = torch.nn.functional.adaptive_avg_pool2d(c_output, (1, 1))
                c_output = c_output.view(c_output.size(0), -1)
                if c_output.size(1) > 1:
                    c_output = c_output.mean(dim=1)
            # Ensure the result is a 1D tensor (one scalar per sample).
            c_output = c_output.squeeze()
        # Store predictions in the appropriate positions.
        self.y_hat[start_idx:start_idx + batch_size] = c_output

    def plot(self):
        # Display a histogram of classifier outputs.
        plt.figure()
        sns.histplot(self.y_hat.cpu().numpy(), stat='proportion', bins=20)
        plt.title("Classifier Output Distribution")
        plt.xlabel("Prediction")
        plt.ylabel("Proportion")
        plt.show()

    def plot_clfs(self):
        # Generate a figure with:
        # 1) A KDE plot of the classifier outputs,
        # 2) A KDE plot of the absolute difference |0.5 - prediction|,
        # 3) A bar chart of the computed hubris value.
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
        # Reset predictions and hubris metric.
        self.y_hat = torch.zeros(self.dataset_size, dtype=torch.float)
        self.hubris.reset()
