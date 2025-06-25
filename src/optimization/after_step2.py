import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from captum.attr import GradientShap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap

from src.data_loaders import load_dataset
from src.utils.checkpoint import construct_classifier_from_checkpoint

# -----------------------------------------------------------------------------
# Dataset config
# -----------------------------------------------------------------------------
# `shape` is (C, H, W) – matches what load_clf() expects.
# Tile size is derived from the last two entries (H, W).

DATASET_INFO: Dict[str, Dict[str, Tuple]] = {
    "mnist":         {"shape": (1, 28, 28),  "mean": (0.5,),               "std": (0.5,)},
    "fashion-mnist": {"shape": (1, 28, 28),  "mean": (0.5,),               "std": (0.5,)},
    "cifar10":       {"shape": (3, 32, 32),  "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "stl10":         {"shape": (3, 96, 96),  "mean": (0.5, 0.5, 0.5),       "std": (0.5, 0.5, 0.5)},
    "chestxray":     {"shape": (1, 224, 224),"mean": (0.5,),               "std": (0.5,)},
    "imagenet":      {"shape": (3, 224,224), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}

BATCH_SIZE = 128
MARGIN_THR = 0.1
RNG        = 0

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sprite → UMAP+GMM → prototypes")
    p.add_argument("sprite", help="Sprite PNG (tiled thumbnails)")
    p.add_argument("--checkpoint", "-c", required=True, help="Classifier checkpoint directory")
    p.add_argument("--dataset", "-d", choices=list(DATASET_INFO), default="fashion-mnist")
    p.add_argument("--samples", "-n", type=int, default=200)
    p.add_argument("--clusters", "-k", type=int, default=8)
    p.add_argument("--out", "-o", default="step2_outputs")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def extract_embeddings(model: nn.Module, data: torch.Tensor, device: torch.device) -> np.ndarray:
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    feat_layer = layers[-2] if len(layers) >= 2 else [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]
    feats = []
    def _hook(_, __, out):
        feats.append(out.detach().cpu().reshape(out.size(0), -1).numpy())
    h = feat_layer.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for batch in torch.split(data.to(device), BATCH_SIZE):
            _ = model(batch)
    h.remove()
    return np.vstack(feats)


def plot_umap(Z: np.ndarray, idxs: list[int], out_dir: Path, name: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c="gray", s=15, alpha=0.3)
    plt.scatter(Z[idxs, 0], Z[idxs, 1], c="red", marker="x", s=100)
    plt.tight_layout()
    plt.savefig(out_dir / f"umap_{name}.png", dpi=150)
    plt.close()


def save_heatmaps(protos: torch.Tensor, clf: nn.Module, device: torch.device, out_dir: Path, c_in: int):
    gs = GradientShap(clf)
    base = torch.full_like(protos, -1.0).to(device)
    for i, x in enumerate(protos):
        x = x.unsqueeze(0).to(device)
        attr = gs.attribute(x, baselines=base[i:i+1], target=1, n_samples=50)
        A   = attr.squeeze(0).cpu().numpy()
        M   = np.abs(A.mean(0)).max() + 1e-8
        img = x.squeeze(0).cpu().permute(1,2,0).numpy()
        H,W = img.shape[:2]
        fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
        ax.axis("off")
        ax.imshow(img[:,:,0] if c_in==1 else img, cmap=None, interpolation="nearest")
        ax.imshow(A.mean(0), cmap="bwr", vmin=-M, vmax=M, alpha=0.6)
        fig.savefig(out_dir/f"proto_{i:02d}_heatmap.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = DATASET_INFO[args.dataset]
    _, tile_h, tile_w = cfg["shape"]

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    clf, *_ = construct_classifier_from_checkpoint(args.checkpoint, device=device)
    c_in    = next(clf.parameters()).shape[1]

    ds, _, _ = load_dataset(args.dataset, os.getenv("FILESDIR", "./data"), pos=None, neg=None, split="test")
    X_test   = torch.stack([ds[i][0] for i in range(len(ds))])

    mode  = "RGB" if c_in==3 else "L"
    sheet = Image.open(args.sprite).convert(mode)
    W,H   = sheet.size
    cols, rows = W//tile_w, H//tile_h
    assert cols*tile_w==W and rows*tile_h==H, "sprite mismatch"

    tfm = T.Compose([T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    tiles=[tfm(sheet.crop((c*tile_w, r*tile_h, (c+1)*tile_w, (r+1)*tile_h)))
           for r in range(rows) for c in range(cols)]
    all_imgs = torch.stack(tiles)

    probs=[]; clf.eval()
    with torch.no_grad():
        for b in torch.split(all_imgs,BATCH_SIZE):
            out = clf(b.to(device))
            p = torch.sigmoid(out) if out.dim()==1 or out.shape[1]==1 else F.softmax(out,1)[:,1]
            probs.append(p.cpu())
    probs   = torch.cat(probs)
    margins = (probs-0.5).abs()
    mask    = margins < MARGIN_THR
    border_all = all_imgs[mask]
    if not len(border_all):
        raise RuntimeError("No borderline samples; increase margin.")
    idx     = torch.argsort(margins[mask])[:args.samples]
    border  = border_all[idx].to(device)

    E_border = extract_embeddings(clf, border, device)
    E_test   = extract_embeddings(clf, X_test, device)
    reducer  = umap.UMAP(15,0.1,2,RNG)
    Z_all    = reducer.fit_transform(np.vstack([E_test,E_border]))
    Z_border = Z_all[len(E_test):]

    gmm   = GaussianMixture(args.clusters, covariance_type="full", init_params="random", random_state=RNG)
    labs  = gmm.fit_predict(Z_border)

    prot_idx=[]
    for c in range(args.clusters):
        cls=np.where(labs==c)[0]
        if not len(cls):
            continue
        D = np.linalg.norm(E_border[cls][:,None]-E_border[cls][None],axis=-1)
        prot_idx.append(int(cls[D.sum(1).argmin()]))

    base_idx = random.sample(range(len(Z_border)), len(prot_idx))
    plot_umap(Z_border, base_idx, out_dir, "baseline")
    plot_umap(Z_border, prot_idx, out_dir, "prototypes")

    save_heatmaps(border[prot_idx], clf, device, out_dir, c_in)
    print("done", out_dir.resolve())

if __name__=="__main__":
    main()
