#!/usr/bin/env python3
import argparse, json, math, statistics, pickle
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from src.gan import construct_gan
from src.classifier import construct_classifier
from src.utils.checkpoint import construct_classifier_from_checkpoint

DATASET_SPECS = {
    "cifar10":       (3, 32, 32),
    "cifar100":      (3, 32, 32),
    "stl10":         (3, 96, 96),
    "imagenet":      (3, 224, 224),
    "fashion-mnist": (1, 28, 28),
    "mnist":         (1, 28, 28),
    "chest-xray":    (3, 128, 128),
}

def find_config_dir(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "config.json").is_file():
            return p
    raise FileNotFoundError

def load_generator(gen_dir: Path, dataset: str, device):
    ckpt = torch.load(gen_dir / "generator.pth", map_location=device)
    if ckpt.get("meta", {}).get("model_args"):
        g_args = ckpt["meta"]["model_args"]
    elif ckpt.get("config", {}).get("model"):
        g_args = ckpt["config"]["model"]
    else:
        g_args = json.loads((find_config_dir(gen_dir)/"config.json").read_text())["model"]
    g_args.setdefault("img_size", DATASET_SPECS[dataset])
    img_sz = tuple(g_args["img_size"])
    G, _ = construct_gan(g_args, img_sz, device)[:2]

    state, model_sd = ckpt.get("state", ckpt), G.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    G.load_state_dict(filtered, strict=False)
    return G.eval()

def load_classifier(clf_dir: Path, dataset: str, device):
    try:
        C, *_ = construct_classifier_from_checkpoint(str(find_config_dir(clf_dir)), device)
        return C.eval()
    except FileNotFoundError:
        C_, H, W = DATASET_SPECS[dataset]
        C = construct_classifier({"type":"mlp","n_classes":2,"img_size":(C_,H,W),"hidden_dim":512}, device)
        path = clf_dir/"classifier.pth"
        try:
            ckpt = torch.load(path, map_location=device)
        except (pickle.UnpicklingError, AttributeError):
            ckpt = torch.load(path, map_location=device, weights_only=False)
        C.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        return C.eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir"); ap.add_argument("clf_dir")
    ap.add_argument("--dataset", required=True, choices=DATASET_SPECS)
    ap.add_argument("--num", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    args = ap.parse_args()

    device = torch.device(args.device)
    G = load_generator(Path(args.gen_dir), args.dataset, device)
    C = load_classifier(Path(args.clf_dir), args.dataset, device)

    z = torch.randn(args.num, getattr(G, "z_dim", 128), device=device)
    imgs = []
    for i in tqdm(range(0, args.num, args.batch), desc="Generating"):
        with torch.no_grad():
            imgs.append(G(z[i:i+args.batch]).clamp(-1,1).add(1).div(2).cpu())
    imgs = torch.cat(imgs)[:args.num]

    scored, acds = [], []
    for img in tqdm(imgs, desc="Scoring"):
        with torch.no_grad():
            p = torch.softmax(C(img.unsqueeze(0).to(device)), -1)[0,1].item()
        acd = abs(p-0.5)
        scored.append((acd, img)); acds.append(acd)

    scored.sort(key=lambda t: t[0])
    sorted_imgs = [img for _, img in scored]

    rows = math.ceil(args.num/args.cols)
    pad  = rows*args.cols-args.num
    if pad:
        sorted_imgs += [torch.zeros_like(sorted_imgs[0])]*pad

    sheet = torch.cat([torch.cat(sorted_imgs[c*rows:(c+1)*rows],1)
                       for c in range(args.cols)], 2)

    out_path = Path(args.out).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)   # minimal change
    save_image(sheet, str(out_path))
    print("Saved", out_path)

if __name__ == "__main__":
    main()
