import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

import umap.umap_ as umap
import matplotlib.pyplot as plt
from captum.attr import GradientShap

from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.data_loaders import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Steps 2–4: slice sprite → margin‐filter → UMAP+GMM → medoids → heatmaps"
    )
    p.add_argument("--sprite",        required=True, help="Sprite PNG")
    p.add_argument("--cell-h",        type=int, required=True, help="Tile height")
    p.add_argument("--cell-w",        type=int, required=True, help="Tile width")
    p.add_argument("--cols",          type=int, default=None,
                   help="Tiles per row (default=img_w//cell_w)")
    p.add_argument("--n-samples",     type=int, default=200, help="How many samples")
    p.add_argument("--pick",          choices=["random","middle"], default="middle",
                   help="random or middle (closest to 0.5 conf)")
    p.add_argument("--clf-checkpoint", required=True, help="Classifier checkpoint dir")
    p.add_argument("--dataset",       choices=["mnist","fashion-mnist","cifar10","stl10"],
                   help="Dataset for UMAP context")
    p.add_argument("--split",         default="test", help="Which split for context")
    p.add_argument("--gmm-clusters",  type=int, default=8, help="Number of GMM clusters")
    p.add_argument("--batch-size",    type=int, default=128, help="Batch size")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",       default="step2_outputs", help="Where to save outputs")
    return p.parse_args()


def extract_embeddings(model: nn.Module, data: torch.Tensor,
                       batch_size: int, device: torch.device) -> np.ndarray:
    # Hook the penultimate layer (last Linear if available, else last Conv2d)
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(layers) >= 2:
        feat_layer = layers[-2]
    else:
        from torch.nn import Conv2d
        feat_layer = [m for m in model.modules() if isinstance(m, Conv2d)][-1]

    feats = []
    def hook(_, __, out):
        feats.append(out.detach().cpu().reshape(out.size(0), -1).numpy())
    h = feat_layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for batch in torch.split(data.to(device), batch_size):
            _ = model(batch)
    h.remove()
    return np.vstack(feats)


def plot_umap(Z: np.ndarray, idxs: list, out_dir: Path, name: str):
    plt.figure(figsize=(6,6))
    plt.scatter(Z[:,0], Z[:,1],
                c='gray', s=15, alpha=0.3, label='borderline pts')
    pts = Z[idxs]
    plt.scatter(pts[:,0], pts[:,1],
                c='red', marker='x', s=100, label='prototypes')
    plt.legend(loc='best')
    plt.title(f"UMAP + GMM {name}")
    plt.savefig(out_dir / f"umap_{name}.png", dpi=150)
    plt.close()


def save_heatmaps(protos: torch.Tensor, clf: nn.Module,
                  device: torch.device, out_dir: Path, C_in: int):
    gs = GradientShap(clf)
    # baseline = pure black (normalized to -1)
    baselines = torch.full_like(protos, fill_value=-1.0).to(device)

    for i, x in enumerate(protos):
        x = x.unsqueeze(0).to(device)
        # target=1 makes sure we get attribution for the positive class
        attr = gs.attribute(
            x,
            baselines=baselines[i:i+1],
            target=1,
            n_samples=50
        )
        A = attr.squeeze(0).cpu().numpy()  # C×H×W
        M = np.max(np.abs(A.mean(0))) + 1e-8

        # convert back to image for display
        img = x.squeeze(0).cpu().permute(1,2,0).numpy()
        H, W = img.shape[:2]
        dpi = 100
        fig, ax = plt.subplots(
            1,1,
            figsize=(W/dpi, H/dpi),
            dpi=dpi,
            frameon=False
        )
        ax.set_axis_off()

        # show base image
        if C_in == 1:
            ax.imshow(img[:,:,0], cmap='gray', interpolation='nearest')
        else:
            ax.imshow(img, interpolation='nearest')

        # overlay attribution heatmap
        im = ax.imshow(
            A.mean(0),
            cmap='bwr',
            alpha=0.6,
            vmin=-M, vmax=+M,
            interpolation='nearest'
        )

        # add colorbar legend
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("GradientShap attribution", rotation=270, labelpad=15)

        plt.tight_layout()
        fig.savefig(
            out_dir / f"proto_{i:02d}_heatmap.png",
            dpi=dpi, bbox_inches='tight', pad_inches=0
        )
        plt.close(fig)


def main():
    args    = parse_args()
    device  = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Load classifier
    clf, *_ = construct_classifier_from_checkpoint(args.clf_checkpoint, device=device)
    clf.eval()
    C_in    = next(clf.parameters()).shape[1]

    # 2) Optional test‐set for UMAP context
    if args.dataset:
        ds, _, _ = load_dataset(
            args.dataset,
            os.getenv("FILESDIR","./data"),
            pos=None, neg=None,
            split=args.split
        )
        X_test = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
        print(f"Loaded {len(X_test)} test samples for context")
    else:
        X_test = None

    # 3) Slice sprite into tiles
    mode = 'RGB' if C_in==3 else 'L'
    sheet= Image.open(args.sprite).convert(mode)
    W,H  = sheet.size
    cols = args.cols or (W // args.cell_w)
    rows = H // args.cell_h

    mean = [0.5]*C_in; std=[0.5]*C_in
    tfm  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    tiles= []
    for r in range(rows):
        for c in range(cols):
            l,u = c*args.cell_w, r*args.cell_h
            crop= sheet.crop((l,u,l+args.cell_w, u+args.cell_h))
            tiles.append(tfm(crop))
    all_imgs = torch.stack(tiles)
    N = len(all_imgs)
    print(f"{N} tiles loaded")

    # 4) Margin filter: keep |p − 0.5| < 0.1
    probs=[]
    with torch.no_grad():
        for b in torch.split(all_imgs, args.batch_size):
            out = clf(b.to(device))
            p   = (torch.sigmoid(out) if out.dim()==1 or out.shape[1]==1
                   else F.softmax(out,1)[:,1]).cpu()
            probs.append(p)
    probs      = torch.cat(probs)
    margins    = torch.abs(probs - 0.5)
    mask       = margins < 0.1
    border_all = all_imgs[mask]
    M = len(border_all)
    print(f"{M} borderline samples (|p-0.5|<0.1)")

    # 5) Pick samples (random or most‐borderline)
    if args.pick == "random":
        idxs = random.sample(range(M), min(args.n_samples, M))
    else:
        idxs = torch.argsort(margins[mask])[:args.n_samples].tolist()
    border = border_all[idxs].to(device)
    print(f"{len(idxs)} selected via '{args.pick}'")

    # 6) Extract embeddings for UMAP
    E_border = extract_embeddings(clf, border, args.batch_size, device)
    if X_test is not None:
        E_test = extract_embeddings(clf, X_test, args.batch_size, device)

    # 7) Joint UMAP
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1,
        n_components=2, random_state=0, n_jobs=1
    )
    if X_test is not None:
        all_E    = np.vstack([E_test, E_border])
        reducer.fit(all_E)
        Z_all    = reducer.transform(all_E)
        Z_border = Z_all[len(E_test):]
    else:
        Z_border = reducer.fit_transform(E_border)

    # 8) GMM + clustering metrics
    gmm    = GaussianMixture(
        n_components=args.gmm_clusters,
        covariance_type="full",
        init_params="random",
        random_state=0
    )
    labels = gmm.fit_predict(Z_border)
    print("silhouette:", silhouette_score(Z_border, labels))
    print("DB index: ", davies_bouldin_score(Z_border, labels))

    # 9) Find medoid indices per cluster
    prototypes_idx = []
    for c in range(args.gmm_clusters):
        cls_idx = np.where(labels == c)[0]
        if not len(cls_idx):
            continue
        Ec = E_border[cls_idx]
        D  = np.linalg.norm(Ec[:,None] - Ec[None,:], axis=-1)
        prototypes_idx.append(int(cls_idx[D.sum(1).argmin()]))
    baseline_idx = random.sample(range(Z_border.shape[0]), len(prototypes_idx))

    plot_umap(Z_border, baseline_idx, out_dir, "baseline")
    plot_umap(Z_border, prototypes_idx, out_dir, "prototypes")

    # 10) Generate and save heatmaps for each prototype
    prototypes = border[prototypes_idx]
    save_heatmaps(prototypes, clf, device, out_dir, C_in)

    print("Done – outputs in", out_dir.resolve())


if __name__ == "__main__":
    main()
