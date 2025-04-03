"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code adapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import math
import numpy as np
import torch
from scipy import linalg

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def load_statistics_from_path(path):
    """
    Load statistics from  npz file

    Params:
    -- path  : Path to .npz file

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    with np.load(path) as f:
        m, s = f['mu'][:], f['sigma'][:]

    return m, s



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.
    The FID between two multivariate Gaussians X_1 ~ N(mu1, sigma1)
    and X_2 ~ N(mu2, sigma2) is:

         ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))

    Regularization with eps is added to the diagonals of the covariance
    matrices to handle numerical instability and potential model collapse.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

    diff = mu1 - mu2

    # Regularize the covariance matrices by adding eps to the diagonal.
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1_reg = sigma1 + offset
    sigma2_reg = sigma2 + offset

    # Check for collapse: if sigma1 is near zero, it may indicate collapse.
    if np.linalg.norm(sigma1) < eps:
        print("Warning: sigma1 is nearly zero. Model collapse suspected. Returning high FID.")
        return np.inf

    covmean, _ = linalg.sqrtm(sigma1_reg.dot(sigma2_reg), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding eps={eps} to diagonal of cov estimates"
        print(msg)
        covmean = linalg.sqrtm(sigma1_reg.dot(sigma2_reg))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid_value = diff.dot(diff) + np.trace(sigma1_reg) + np.trace(sigma2_reg) - 2 * tr_covmean
    return fid_value


def get_activations(images, feature_map_fn, batch_size=64, dims=2048, device='cpu'):
    """Calculates the activations of layer returned by feature_map_fn.

    Params:
    -- images           : Tensor of images (N images, C, H, W)
    -- feature_map_fn   : Function used to obtain layer
    -- batch_size       : Batch size of images for the model to process at once.
    -- dims             : Dimensionality of features returned by Inception (or other Classifier used)
    -- device           : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if batch_size > len(images):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(images)

    pred_arr = np.empty((len(images), dims))

    start_idx = 0
    num_batches = math.ceil(len(images) / batch_size)

    for _ in tqdm(range(num_batches)):
        # batch dim = (batch size, 3, 4, 4)
        batch = images[start_idx:start_idx +
                       min(batch_size, len(images) - start_idx)].to(device)

        with torch.no_grad():
            pred = feature_map_fn(
                batch, start_idx, batch.shape[0])

        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(images, feature_map_fn, batch_size=64, dims=2048, device='cpu'):
    """Calculation of the statistics used by the FID.

    Params:
    -- images           : Tensor of images (N images, C, H, W)
    -- feature_map_fn   : Function used to obtain layer
    -- batch_size       : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims             : Dimensionality of features returned by feature_map_fn
    -- device           : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, feature_map_fn, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations_dataloader(dataloader, feature_map_fn, dims=2048, device='cpu'):
    """Calculates the activations of layer returned by feature_map_fn using images from a DataLoader.

    Params:
    -- dataloader       : PyTorch Dataloader
    -- feature_map_fn   : Function used to obtain layer
    -- dims             : Dimensionality of features returned by feature_map_fn
    -- device           : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    pred_arr = np.empty((len(dataloader.dataset), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch[0].to(device)

        with torch.no_grad():
            pred = feature_map_fn(batch, start_idx, batch.shape[0])

        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics_dataloader(dataloader, feature_map_fn, dims=2048, device='cpu'):
    """Calculation of the statistics used by the FID.

    Params:
    -- dataloader       : PyTorch Dataloader
    -- feature_map_fn   : Function used to obtain layer
    -- dims             : Dimensionality of features returned by feature_map_fn
    -- device           : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_dataloader(dataloader, feature_map_fn, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
