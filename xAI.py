import os
import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF

from PIL import Image
from captum.attr import GradientShap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
import matplotlib.pyplot as plt

from src.data_loaders import load_dataset
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.classifier import construct_classifier

# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
DATASET_INFO = {
    "mnist":         {"size": (28,  28), "mean": (0.5,),             "std": (0.5,)             },
    "fashion-mnist": {"size": (28,  28), "mean": (0.5,),             "std": (0.5,)             },
    "cifar10":       {"size": (32,  32), "mean": (0.485,0.456,0.406),"std": (0.229,0.224,0.225)},
    "stl10":         {"size": (96,  96), "mean": (0.5,0.5,0.5),       "std": (0.5,0.5,0.5)      },
    "chest-xray":    {"size": (128,128), "mean": (0.5,0.5,0.5),       "std": (0.5,0.5,0.5)      },
    "imagenet":      {"size": (224,224), "mean": (0.485,0.456,0.406),"std": (0.229,0.224,0.225)},
}

BATCH_SIZE  = 128
MARGIN_THR  = 0.1
SEED        = 0

# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sprite",         required=True,  help="Canvas with generated images")
    p.add_argument("--clf-checkpoint", required=True,  help="Path to classifier checkpoint dir")
    p.add_argument("--dataset",        required=True,  choices=DATASET_INFO.keys())
    p.add_argument("--n-samples",      type=int, default=200)
    p.add_argument("--gmm-clusters",   type=int, default=8)
    p.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",        default="step2_outputs")
    return p.parse_args()

# ------------------------------------------------------------------
#  Helper: load (or reconstruct) classifier
# ------------------------------------------------------------------
def load_clf(path: Path, dataset: str, device: torch.device):
    try:
        clf, *_ = construct_classifier_from_checkpoint(path, device=device)
        return clf
    except Exception:
        ctype = path.name.split("-")[0]
        nf = int(path.name.split("-")[1]) if "-" in path.name and path.name.split("-")[1].isdigit() else 512
        C = 1 if dataset in {"mnist", "fashion-mnist", "chest-xray"} else 3
        H, W = DATASET_INFO[dataset]["size"]
        params = {"type": ctype, "n_classes": 2, "img_size": (C, H, W), "nf": nf}
        clf = construct_classifier(params, device=device)
        state = torch.load(path / "classifier.pth", map_location=device)
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        clf.load_state_dict(state, strict=False)
        return clf

# ------------------------------------------------------------------
#  Helper: feature extraction
# ------------------------------------------------------------------
def extract_embeddings(model: nn.Module,
                       data:  torch.Tensor,
                       batch_size: int,
                       device: torch.device) -> np.ndarray:
    # penultimate Linear or last Conv2d
    lin_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(lin_layers) >= 2:
        feat_layer = lin_layers[-2]
    else:
        from torch.nn import Conv2d
        feat_layer = [m for m in model.modules() if isinstance(m, Conv2d)][-1]

    feats = []
    h = feat_layer.register_forward_hook(
        lambda _, __, out: feats.append(out.detach().cpu().reshape(out.size(0), -1).numpy())
    )
    model.eval()
    with torch.no_grad():
        for b in torch.split(data.to(device), batch_size):
            _ = model(b)
    h.remove()
    return np.vstack(feats) if feats else np.zeros((0, 1))

# ------------------------------------------------------------------
#  Helper: UMAP scatter
# ------------------------------------------------------------------
def plot_umap(Z: np.ndarray, idxs: np.ndarray, out: Path, name: str):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(Z[:, 0], Z[:, 1], c="gray", s=15, alpha=0.3, label="borderline")
    pts = Z[idxs]
    plt.scatter(pts[:, 0], pts[:, 1], c="r", marker="x", s=80, label="prototypes")
    plt.legend(loc="upper right", frameon=False)
    plt.axis("off")
    plt.savefig(out / f"umap_{name}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

# ------------------------------------------------------------------
#  Helper: GradientShap heat-map + original sprite
# ------------------------------------------------------------------
def save_heatmaps(protos: torch.Tensor,
                  clf:    nn.Module,
                  device: torch.device,
                  out:    Path,
                  c_in:   int) -> None:
    gs = GradientShap(clf)
    baselines = torch.full_like(protos, -1).to(device)

    for i, x in enumerate(protos):
        x_ = x.unsqueeze(0).to(device)
        attr = gs.attribute(x_, baselines=baselines[i:i+1],
                            target=1, n_samples=50).squeeze(0).cpu().numpy()  # C×H×W
        sal = attr.mean(0)                                    # H×W
        vmax = np.abs(sal).max() + 1e-8

        img = x_.squeeze(0).cpu().permute(1, 2, 0).numpy()    # H×W×C
        H, W = sal.shape

        # upscale if long side <256 px
        target = 256
        if max(H, W) < target:
            scale  = int(np.ceil(target / max(H, W)))
            new_sz = (H * scale, W * scale)
            # sprite – nearest neighbour
            img_t = torch.from_numpy(img).permute(2, 0, 1)
            img_t = TF.resize(img_t, new_sz,
                             interpolation=TF.InterpolationMode.NEAREST)
            img   = img_t.permute(1, 2, 0).numpy()
            # saliency – bicubic
            sal_t = torch.from_numpy(sal).unsqueeze(0)
            sal_t = TF.resize(sal_t, new_sz,
                             interpolation=TF.InterpolationMode.BICUBIC)
            sal   = sal_t.squeeze(0).numpy()
            H, W  = sal.shape

        # ---------------- original sprite (grayscale) ----------------
        if c_in == 1 or img.shape[2] == 1:
            gray = img.squeeze(-1)
        else:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        plt.imsave(out / f"proto_{i:02d}_orig.png", gray, cmap="gray")

        # ---------------- heat-map overlay ---------------------------
        dpi = 100
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax.axis("off")
        ax.imshow(img[:, :, 0] if c_in == 1 else img,
                  cmap="gray" if c_in == 1 else None)
        hm = ax.imshow(sal, cmap="bwr", alpha=0.6, vmin=-vmax, vmax=vmax)
        cbar = plt.colorbar(hm, ax=ax, orientation="horizontal",
                            pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("red: ↑ P(class 1) blue: ↓", fontsize=7)

        plt.savefig(out / f"proto_{i:02d}_heatmap.png",
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)

# ------------------------------------------------------------------
#  Main routine
# ------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) classifier
    clf = load_clf(Path(args.clf_checkpoint), args.dataset, device)
    clf.eval()
    c_in = next(clf.parameters()).shape[1]

    # 2) full test set (context)
    ds, _, _ = load_dataset(args.dataset,
                            os.getenv("FILESDIR", "./data"),
                            None, None)
    Xc = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)

    # 3) sprite -> tensor list
    spec = DATASET_INFO[args.dataset]
    th, tw = spec["size"]; mean, std = spec["mean"], spec["std"]
    spr = Image.open(args.sprite).convert("RGB" if c_in == 3 else "L")
    W, H = spr.size; cols, rows = W // tw, H // th
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    tiles = [tfm(spr.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)))
             for r in range(rows) for c in range(cols)]
    all_imgs = torch.stack(tiles)

    # 4) borderline sample selection
    ps = []
    with torch.no_grad():
        for b in torch.split(all_imgs, args.batch_size):
            logits = clf(b.to(device))
            prob = (torch.sigmoid(logits) if logits.dim() == 1 or logits.shape[1] == 1
                    else F.softmax(logits, 1)[:, 1])
            ps.append(prob.cpu())
    ps = torch.cat(ps)
    margins = (ps - 0.5).abs()
    mask = margins < MARGIN_THR
    idxs = (torch.argsort(margins)[:args.n_samples] if mask.sum() == 0
            else torch.argsort(margins[mask])[:args.n_samples].long())
    border = all_imgs[idxs].to(device)

    # 5) embeddings + UMAP
    Eb = extract_embeddings(clf, border, args.batch_size, device)
    Et = extract_embeddings(clf, Xc, args.batch_size, device)
    Zall = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED) \
             .fit_transform(np.vstack([Et, Eb]))
    Zb = Zall[len(Et):]

    # 6) k-medoids via GMM
    gmm = GaussianMixture(n_components=args.gmm_clusters,
                          covariance_type="full",
                          init_params="random",
                          random_state=SEED)
    labs = gmm.fit_predict(Zb)
    print("Silhouette:", silhouette_score(Zb, labs))
    print("Davies–Bouldin:", davies_bouldin_score(Zb, labs))

    prot = []
    for k in range(args.gmm_clusters):
        cis = np.where(labs == k)[0]
        if cis.size:
            D = np.linalg.norm(Eb[cis][:, None] - Eb[cis][None, :], axis=-1)
            prot.append(int(cis[D.sum(1).argmin()]))

    base = random.sample(range(len(Zb)), len(prot))

    # 7) plots + heat-maps
    plot_umap(Zb, base, out, "baseline")
    plot_umap(Zb, prot, out, "prototypes")
    save_heatmaps(border[prot], clf, device, out, c_in)

    print("Done → outputs in", out.resolve())


if __name__ == "__main__":
    main()
