#!/usr/bin/env python
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from captum.attr import GradientShap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap.umap_ as umap
import matplotlib.pyplot as plt

from src.data_loaders import load_dataset
from src.utils.checkpoint import construct_classifier_from_checkpoint
from src.classifier import construct_classifier

DATASET_INFO = {
    "mnist":         {"size": (28,  28), "mean": (0.5,),                   "std": (0.5,)                   },
    "fashion-mnist": {"size": (28,  28), "mean": (0.5,),                   "std": (0.5,)                   },
    "cifar10":       {"size": (32,  32), "mean": (0.485,0.456,0.406),       "std": (0.229,0.224,0.225)      },
    "stl10":         {"size": (96,  96), "mean": (0.5,0.5,0.5),             "std": (0.5,0.5,0.5)            },
    "chest-xray":    {"size": (128,128), "mean": (0.5,0.5,0.5),             "std": (0.5,0.5,0.5)            },
    "imagenet":      {"size": (224,224), "mean": (0.485,0.456,0.406),       "std": (0.229,0.224,0.225)      },
}

BATCH_SIZE = 128
MARGIN_THR = 0.1
SEED       = 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sprite",         required=True)
    p.add_argument("--clf-checkpoint", required=True)
    p.add_argument("--dataset",        required=True, choices=DATASET_INFO.keys())
    p.add_argument("--n-samples",      type=int, default=200)
    p.add_argument("--gmm-clusters",   type=int, default=8)
    p.add_argument("--batch-size",     type=int, default=BATCH_SIZE)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",        default="step2_outputs")
    return p.parse_args()

def load_clf(path: Path, dataset: str, device: torch.device):
    try:
        clf, *_ = construct_classifier_from_checkpoint(path, device=device)
        return clf
    except Exception:
        ctype = path.name.split("-")[0]
        nf    = int(path.name.split("-")[1]) if "-" in path.name and path.name.split("-")[1].isdigit() else 512
        C = 1 if dataset in {"mnist","fashion-mnist","chest-xray"} else 3
        H,W = DATASET_INFO[dataset]["size"]
        params = {"type":ctype, "n_classes":2, "img_size":(C,H,W), "nf":nf}
        clf = construct_classifier(params, device=device)
        state = torch.load(path/"classifier.pth", map_location=device)
        state = {k.replace("backbone.",""):v for k,v in state.items()}
        clf.load_state_dict(state, strict=False)
        return clf

def extract_embeddings(model, data, batch_size, device):
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(layers)>=2:
        feat_layer = layers[-2]
    else:
        from torch.nn import Conv2d
        feat_layer = [m for m in model.modules() if isinstance(m,Conv2d)][-1]
    feats=[]
    def hook(_,__,out):
        o=out.detach().cpu()
        if o.numel()>0:
            feats.append(o.reshape(o.size(0),-1).numpy())
    h = feat_layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for b in torch.split(data.to(device), batch_size):
            _=model(b)
    h.remove()
    return np.vstack(feats) if feats else np.zeros((0,1))

def plot_umap(Z, idxs, out: Path, name: str):
    plt.figure(figsize=(6,6))
    # background points
    plt.scatter(Z[:,0], Z[:,1], c="gray", s=15, alpha=0.3, label="borderline")
    # prototype points
    pts = Z[idxs]
    plt.scatter(pts[:,0], pts[:,1], c="r", marker="x", s=80, label="prototypes")
    plt.legend(loc="upper right", frameon=False)
    plt.axis("off")
    plt.savefig(out / f"umap_{name}.png", dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_heatmaps(protos,clf,device,out: Path,c_in: int):
    gs=GradientShap(clf)
    baselines=torch.full_like(protos,-1).to(device)
    for i,x in enumerate(protos):
        x_=x.unsqueeze(0).to(device)
        A=gs.attribute(x_,baselines=baselines[i:i+1],target=1,n_samples=50).squeeze(0).cpu().numpy()
        M=np.max(np.abs(A.mean(0)))+1e-8
        img=x_.squeeze(0).cpu().permute(1,2,0).numpy()
        H,W=img.shape[:2]; dpi=100
        fig,ax=plt.subplots(figsize=(W/dpi,H/dpi),dpi=dpi,frameon=False)
        ax.axis("off")
        ax.imshow(img[:,:,0] if c_in==1 else img, cmap="gray" if c_in==1 else None)
        ax.imshow(A.mean(0),cmap="bwr",alpha=0.6,vmin=-M,vmax=M)
        plt.savefig(out / f"proto_{i:02d}_heatmap.png", bbox_inches="tight", pad_inches=0)
        plt.close()

def main():
    args = parse_args()
    device = torch.device(args.device)
    out = Path(args.out_dir); out.mkdir(exist_ok=True, parents=True)

    clf = load_clf(Path(args.clf_checkpoint), args.dataset, device)
    clf.eval()
    c_in = next(clf.parameters()).shape[1]

    # full test set for context
    ds,_,_ = load_dataset(args.dataset, os.getenv("FILESDIR","./data"), None, None)
    Xc = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)

    spec = DATASET_INFO[args.dataset]
    th,tw = spec["size"]; mean,std = spec["mean"],spec["std"]
    spr = Image.open(args.sprite).convert("RGB" if c_in==3 else "L")
    W,H = spr.size; cols,rows = W//tw, H//th
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    tiles = [tfm(spr.crop((c*tw,r*th,(c+1)*tw,(r+1)*th))) for r in range(rows) for c in range(cols)]
    all_imgs = torch.stack(tiles)

    ps=[]
    with torch.no_grad():
        for b in torch.split(all_imgs, args.batch_size):
            outp = clf(b.to(device))
            p = (torch.sigmoid(outp) if outp.dim()==1 or outp.shape[1]==1 else F.softmax(outp,1)[:,1]).cpu()
            ps.append(p)
    ps = torch.cat(ps)
    margins = (ps-0.5).abs()
    mask = margins < MARGIN_THR
    if mask.sum()==0:
        idxs = torch.argsort(margins)[:args.n_samples]
    else:
        idxs = torch.argsort(margins[mask])[:args.n_samples]
        idxs = torch.tensor(np.array(idxs),dtype=torch.long)
    border = all_imgs[idxs].to(device)

    Eb = extract_embeddings(clf, border, args.batch_size, device)
    Et = extract_embeddings(clf, Xc, args.batch_size, device)
    Zall = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit_transform(np.vstack([Et,Eb]))
    Zb = Zall[len(Et):]

    gmm = GaussianMixture(n_components=args.gmm_clusters,
                          covariance_type="full",
                          init_params="random",
                          random_state=SEED)
    labs = gmm.fit_predict(Zb)
    print("Silhouette:", silhouette_score(Zb,labs))
    print("DB index:", davies_bouldin_score(Zb,labs))

    prot=[]
    for k in range(args.gmm_clusters):
        cis = np.where(labs==k)[0]
        if not len(cis): continue
        D = np.linalg.norm(Eb[cis][:,None]-Eb[cis][None,:],axis=-1)
        prot.append(int(cis[D.sum(1).argmin()]))
    base = random.sample(range(len(Zb)), len(prot))

    plot_umap(Zb, base, out, "baseline")
    plot_umap(Zb, prot, out, "prototypes")
    save_heatmaps(border[prot], clf, device, out, c_in)

    print("Done → outputs in", out.resolve())

if __name__=="__main__":
    main()
