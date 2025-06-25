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
        cfg_dir = find_config_dir(gen_dir)
        g_args  = json.loads((cfg_dir / "config.json").read_text())["model"]
    g_args.setdefault("img_size", DATASET_SPECS[dataset])
    img_sz = tuple(g_args["img_size"])
    G, _ = construct_gan(g_args, img_sz, device)[:2]

    state = ckpt.get("state", ckpt)
    model_sd = G.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    print(f"[GEN] loaded {len(filtered)} tensors | "
          f"skipped {len(model_sd)-len(filtered)} missing, "
          f"{len(state)-len(filtered)} size-mismatched/unused")
    G.load_state_dict(filtered, strict=False)
    return G.eval()

def load_classifier(clf_dir: Path, dataset: str, device):
    try:
        cfg_dir = find_config_dir(clf_dir)
        C, *_   = construct_classifier_from_checkpoint(str(cfg_dir), device)
        return C.eval()
    except FileNotFoundError:
        C_, H, W = DATASET_SPECS[dataset]
        params = {"type": "mlp", "n_classes": 2,
                  "img_size": (C_, H, W), "hidden_dim": 512}
        C = construct_classifier(params, device)
        ckpt_path = clf_dir / "classifier.pth"
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except (pickle.UnpicklingError, AttributeError):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        C.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        return C.eval()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("clf_dir")
    ap.add_argument("--dataset", required=True, choices=DATASET_SPECS)
    ap.add_argument("--num",   type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cols",  type=int, default=100)
    ap.add_argument("--out",   default="sheet.png")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    device  = torch.device(args.device)
    G = load_generator(Path(args.gen_dir), args.dataset, device)
    C = load_classifier(Path(args.clf_dir), args.dataset, device)

    z_dim = getattr(G, "z_dim", 128)
    zs    = torch.randn(args.num, z_dim, device=device)

    imgs = []
    for i in tqdm(range(0, args.num, args.batch), desc="Generating"):
        with torch.no_grad():
            imgs.append(G(zs[i:i+args.batch]).clamp(-1, 1).add(1).div(2).cpu())
    imgs = torch.cat(imgs)[: args.num]

    scored = []
    acds   = []
    for img in tqdm(imgs, desc="Scoring"):
        with torch.no_grad():
            p_pos = torch.softmax(C(img.unsqueeze(0).to(device)), -1)[0, 1].item()
        acd = abs(p_pos - 0.5)
        acds.append(acd)
        scored.append((acd, img))

    print(f"ACD mean {statistics.mean(acds):.4f} "
          f"median {statistics.median(acds):.4f} "
          f"<0.1 {(sum(a < 0.1 for a in acds)*100/len(acds)):.1f}%")

    scored.sort(key=lambda t: t[0])
    sorted_imgs = [img for _, img in scored]

    rows = math.ceil(args.num / args.cols)
    pad  = rows * args.cols - args.num
    if pad:
        sorted_imgs += [torch.zeros_like(sorted_imgs[0])] * pad

    cols = [torch.cat(sorted_imgs[c*rows:(c+1)*rows], 1) for c in range(args.cols)]
    sheet = torch.cat(cols, 2)

    out_path = Path(args.out).with_suffix(".png")
    save_image(sheet, str(out_path))
    print("Saved", out_path)

if __name__ == "__main__":
    main()
