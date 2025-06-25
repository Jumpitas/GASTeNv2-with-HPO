#!/usr/bin/env python3
import argparse, json, math, statistics
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
    "chest-xray":    (1, 128, 128),
}

def find_config_dir(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "config.json").is_file():
            return p
    raise FileNotFoundError(f"config.json not found under {start} or its parents")

def load_generator(gen_dir: Path, dataset: str, device: torch.device):
    cfg_dir  = find_config_dir(gen_dir)
    cfg      = json.loads((cfg_dir / "config.json").read_text())
    img_size = DATASET_SPECS[dataset]
    G, _     = construct_gan(cfg["model"], img_size, device)[:2]
    ckpt     = torch.load(gen_dir / "generator.pth", map_location=device, weights_only=False)
    state    = ckpt.get("state", ckpt)
    G.load_state_dict(state)
    return G.eval()

def load_classifier(clf_dir: Path, dataset: str, device: torch.device):
    try:
        cfg_dir = find_config_dir(clf_dir)
        C, *_   = construct_classifier_from_checkpoint(str(cfg_dir), device)
        return C.eval()
    except FileNotFoundError:
        C_, H, W = DATASET_SPECS[dataset]
        params = {
            "type": "mlp",
            "n_classes": 2,
            "img_size": (C_, H, W),
            "hidden_dim": 512
        }
        C = construct_classifier(params, device)
        ckpt = torch.load(clf_dir / "classifier.pth", map_location=device, weights_only=False)
        sd   = ckpt.get("state_dict", ckpt)
        C.load_state_dict(sd, strict=False)
        return C.eval()

def main():
    p = argparse.ArgumentParser(
        description="Generate & sort images by ACD (|p_pos–0.5|) from a trained GAN"
    )
    p.add_argument("gen_dir", help="Directory with generator.pth & config.json")
    p.add_argument("clf_dir", help="Directory with classifier.pth (& config.json)")
    p.add_argument("--dataset", required=True, choices=DATASET_SPECS.keys())
    p.add_argument("--num",   type=int, default=10_000,
                   help="Total number of images to generate")
    p.add_argument("--batch", type=int, default=64,
                   help="Batch size for generation")
    p.add_argument("--cols",  type=int, default=100,
                   help="Number of columns in final sheet")
    p.add_argument("--out",   default="sheet.png",
                   help="Output filename (will append .png if missing)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()

    device = torch.device(args.device)
    gen_dir = Path(args.gen_dir)
    clf_dir = Path(args.clf_dir)

    print("Loading generator…")
    G = load_generator(gen_dir, args.dataset, device)
    print("Loading classifier…")
    C = load_classifier(clf_dir, args.dataset, device)

    # generate latents
    z_dim = getattr(G, "z_dim", 128)
    zs = torch.randn(args.num, z_dim, device=device)

    # forward through G
    imgs = []
    for i in tqdm(range(0, args.num, args.batch), desc="Generating"):
        with torch.no_grad():
            batch = zs[i:i+args.batch]
            out   = G(batch).cpu().clamp(-1,1).add(1).div(2)
        imgs.append(out)
    imgs = torch.cat(imgs, 0)[: args.num]  # (N, C, H, W)

    # score and compute ACD = |p_pos - 0.5|
    scored = []
    acds = []
    for img in tqdm(imgs, desc="Scoring"):
        with torch.no_grad():
            logits = C(img.unsqueeze(0).to(device))
            probs  = torch.softmax(logits, dim=-1).cpu().squeeze(0)
            p_pos  = float(probs[1])
        acd = abs(p_pos - 0.5)
        acds.append(acd)
        scored.append((acd, img))

    # print stats
    mean_acd   = statistics.mean(acds)
    median_acd = statistics.median(acds)
    frac_01    = sum(1 for a in acds if a < 0.1) / len(acds)
    print(f"ACD stats: mean={mean_acd:.4f}, median={median_acd:.4f}, "
          f"fraction(<0.1)={frac_01*100:.1f}%")

    # sort by ACD ascending (closest to boundary first)
    scored.sort(key=lambda x: x[0])
    sorted_imgs = [img for _, img in scored]

    # pad to full grid
    rows = math.ceil(args.num / args.cols)
    pad = rows * args.cols - args.num
    if pad > 0:
        blank = torch.zeros_like(sorted_imgs[0])
        sorted_imgs += [blank]*pad

    # build columns (vertical stacks)
    cols = []
    for c in range(args.cols):
        start = c*rows
        block = sorted_imgs[start:start+rows]
        cols.append(torch.cat(block, dim=1))
    # then build full sheet
    sheet = torch.cat(cols, dim=2)

    # ensure .png
    out_path = Path(args.out)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")

    save_image(sheet, str(out_path))
    print(f"Saved sheet to {out_path}")

if __name__ == "__main__":
    main()
