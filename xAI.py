#!/usr/bin/env python3
from pathlib import Path
import os
import argparse
import random
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
import matplotlib.pyplot as plt
from captum.attr import GradientShap

from src.utils.checkpoint import construct_gan_from_checkpoint, construct_classifier_from_checkpoint
from src.data_loaders import load_dataset

# extract penultimate‐layer embeddings
def extract_embeddings(model, x, batch_size, device):
    # hook into the second‐to‐last Linear (or last Conv2d)
    layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    if len(layers) >= 2:
        feat_layer = layers[-2]
    else:
        from torch.nn import Conv2d
        feat_layer = [m for m in model.modules() if isinstance(m, Conv2d)][-1]

    feats = []
    def hook(_, __, out):
        feats.append(out.detach().cpu().view(out.size(0), -1).numpy())
    h = feat_layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for batch in torch.split(x.to(device), batch_size):
            _ = model(batch)
    h.remove()
    return np.vstack(feats)

def plot_umap(Z, idxs, out_dir, name):
    plt.figure(figsize=(6,6))
    plt.scatter(Z[:,0], Z[:,1], c='gray', s=15, alpha=0.3)
    sel = Z[idxs]
    plt.scatter(sel[:,0], sel[:,1], c='red', marker='x', s=100)
    plt.title(f"UMAP + GMM: {name}")
    plt.savefig(out_dir / f"umap_{name}.png", dpi=150)
    plt.close()

def save_heatmaps(protos, clf, device, out_dir, C_in):
    gs = GradientShap(clf)
    baselines = torch.full_like(protos, -1.0).to(device)
    for i, x in enumerate(protos):
        x = x.unsqueeze(0).to(device)
        attr = gs.attribute(
            x, baselines=baselines[i:i+1], target=1, n_samples=50
        ).squeeze(0).cpu().numpy()  # C×H×W
        # convert to image for display
        img = x.squeeze(0).cpu().permute(1,2,0).numpy()
        H,W = img.shape[:2]
        M = np.max(np.abs(attr.mean(0))) + 1e-8

        fig, ax = plt.subplots(1,1,figsize=(W/100,H/100),dpi=100,frameon=False)
        ax.set_axis_off()
        if C_in==1:
            ax.imshow(img[:,:,0], cmap='gray', interpolation='nearest')
        else:
            ax.imshow(img, interpolation='nearest')
        im = ax.imshow(attr.mean(0), cmap='bwr', alpha=0.6,
                       vmin=-M, vmax=+M, interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("GradientShap attr", rotation=270, labelpad=15)
        fig.savefig(out_dir / f"proto_{i:02d}_heatmap.png",
                    dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def main():
    p = argparse.ArgumentParser(
        description="Generate from GAN → margin filter → UMAP+GMM → prototypes → heatmaps"
    )
    p.add_argument("gan_dir", help="Directory with generator.pth (+ optional config.json)")
    p.add_argument("clf_dir", help="Dir with classifier.pth (+ config.json)")
    p.add_argument("--dataset", required=True,
                   choices=["mnist","fashion-mnist","cifar10","stl10"],
                   help="Dataset for context embedding")
    p.add_argument("--num",    type=int, default=2000,
                   help="How many images to sample from GAN")
    p.add_argument("--batch",  type=int, default=128,
                   help="Batch size for generation & classification")
    p.add_argument("--n-samples", type=int, default=200,
                   help="How many borderline points to keep")
    p.add_argument("--pick",   choices=["random","middle"], default="middle",
                   help="Random or most‐borderline")
    p.add_argument("--gmm-clusters", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="step2_outputs")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load GAN & classifier
    G, D, _, _ = construct_gan_from_checkpoint(args.gan_dir, device=device)
    G.eval()
    clf, *_ = construct_classifier_from_checkpoint(args.clf_dir, device=device)
    clf.eval()
    C_in = next(clf.parameters()).shape[1]

    # 2) optional context dataset embeddings
    ds, _, _ = load_dataset(args.dataset, os.getenv("FILESDIR","./data"),
                             pos=None, neg=None, split="test")
    X_ctx = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    E_ctx = extract_embeddings(clf, X_ctx, args.batch, device)

    # 3) sample noise → images
    zs = torch.randn(args.num, G.z_dim, device=device)
    imgs = []
    with torch.no_grad(), tqdm(total=args.num, desc="Generating") as bar:
        for z in torch.split(zs, args.batch):
            batch = G(z).cpu().clamp(-1,1).add(1).div(2)
            imgs.append(batch)
            bar.update(len(z))
    imgs = torch.cat(imgs, 0)  # [N,C,H,W]

    # 4) margin‐filter
    probs = []
    with torch.no_grad(), tqdm(total=len(imgs), desc="Scoring") as bar:
        for b in torch.split(imgs.to(device), args.batch):
            out = clf(b)
            p = (torch.sigmoid(out) if out.dim()==1 or out.shape[1]==1
                 else F.softmax(out,1)[:,1])
            probs.append(p.cpu())
            bar.update(b.size(0))
    probs = torch.cat(probs)
    margins = torch.abs(probs - 0.5)
    mask = margins < 0.1
    borderline = imgs[mask]
    M = borderline.size(0)
    print(f"{M} borderline samples (|p-0.5|<0.1)")

    # 5) pick subset
    if args.pick=="random":
        idxs = random.sample(range(M), min(args.n_samples, M))
    else:
        idxs = margins[mask].argsort()[:args.n_samples].tolist()
    border = borderline[idxs].to(device)

    # 6) embeddings + UMAP
    E_border = extract_embeddings(clf, border, args.batch, device)
    all_E = np.vstack([E_ctx, E_border])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2, random_state=0, n_jobs=1)
    reducer.fit(all_E)
    Z_all = reducer.transform(all_E)
    Z_border = Z_all[len(E_ctx):]

    # 7) GMM + cluster metrics
    gmm = GaussianMixture(n_components=args.gmm_clusters,
                          covariance_type="full",
                          init_params="random",
                          random_state=0)
    labels = gmm.fit_predict(Z_border)
    print("Silhouette:", silhouette_score(Z_border, labels))
    print("Davies–Bouldin:", davies_bouldin_score(Z_border, labels))

    # 8) medoids per cluster
    medoids = []
    for c in range(args.gmm_clusters):
        idx_c = np.where(labels==c)[0]
        if len(idx_c)==0: continue
        E_c = E_border[idx_c]
        Dmat = np.linalg.norm(E_c[:,None]-E_c[None,:], axis=-1)
        med = idx_c[Dmat.sum(1).argmin()]
        medoids.append(int(med))
    # baselines = random.sample(range(len(E_border)), len(medoids))

    plot_umap(Z_border, medoids, out_dir, "medoids")

    # 9) heatmaps
    protos = border[medoids]
    save_heatmaps(protos, clf, device, out_dir, C_in)

    # 10) save medoid images grid
    grid = torch.cat([protos.cpu().add(-0.5).mul(1.0)], 0)  # [k,C,H,W]
    save_image(grid, out_dir/"medoids.png", nrow=len(medoids), normalize=True)

    print("Done! Outputs in", out_dir)

if __name__=="__main__":
    main()
