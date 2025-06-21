#!/usr/bin/env python3
import os
import glob
import json
import argparse
import torch
import numpy as np
from PIL import Image

from src.utils.checkpoint import construct_gan_from_checkpoint
from src.utils.config import read_config

def find_best_run(root_dir, metrics_filename, metric_key):
    best = {"value": float("inf"), "path": None}
    pattern = os.path.join(root_dir, "*", metrics_filename)
    for metrics_file in glob.glob(pattern):
        run_dir = os.path.dirname(metrics_file)
        try:
            with open(metrics_file, "r") as f:
                m = json.load(f)
            vals = m.get(metric_key)
            if not isinstance(vals, list) or not vals:
                continue
            final = vals[-1]
            if final < best["value"]:
                best["value"] = final
                best["path"]  = run_dir
        except Exception:
            continue
    return best

def main():
    p = argparse.ArgumentParser(
        description="Generate images from the best Step-2 GAN"
    )
    p.add_argument("root",
                   help="Root directory containing all Step-2 run subfolders")
    p.add_argument("--metric-file", default="eval_metrics.json",
                   help="Filename (in each run dir) containing the metrics JSON")
    p.add_argument("--metric-key", default="fid",
                   help="Which key in the JSON to compare (must be a list)")
    p.add_argument("--n-images", type=int, default=10000,
                   help="Total number of images to generate")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size for generation")
    p.add_argument("--out-dir", default="gen_images",
                   help="Where to save the generated PNGs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device to use")
    args = p.parse_args()

    # 1) Find best run by final metric
    best = find_best_run(args.root, args.metric_file, args.metric_key)
    if best["path"] is None:
        print("No valid runs found under", args.root)
        return
    print("Best run:", best["path"])
    print("Final {} = {:.4f}".format(args.metric_key, best["value"]))

    # 2) Load the Gaussian weights file if present
    gauss_files = glob.glob(os.path.join(best["path"], "step-2-best-gauss-*.json"))
    if gauss_files:
        with open(gauss_files[0], "r") as f:
            weight = json.load(f)
        print("Loaded step-2 Gaussian weights:", weight)
    else:
        print("No step-2-best-gauss-*.json found in", best["path"])

    # 3) Read the GAN config to get z_dim
    cfgs = glob.glob(os.path.join(best["path"], "*.yaml"))
    if not cfgs:
        print("Could not find a .yaml config in", best["path"])
        return
    config = read_config(cfgs[0])
    z_dim  = config["model"]["z_dim"]

    # 4) Reconstruct the generator from checkpoint
    device = torch.device(args.device)
    G, D, _, _ = construct_gan_from_checkpoint(best["path"], device=device)
    G.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # 5) Sample and save images
    n_done = 0
    with torch.no_grad():
        while n_done < args.n_images:
            this_batch = min(args.batch_size, args.n_images - n_done)
            z = torch.randn(this_batch, z_dim, device=device)
            imgs = G(z)                    # outputs in [-1,1]
            imgs = (imgs + 1) * 0.5        # scale to [0,1]
            arrs = imgs.clamp(0,1).cpu().numpy()

            for arr in arrs:
                # convert to H×W×C uint8
                img = (arr.transpose(1,2,0) * 255).astype(np.uint8)
                if img.shape[2] == 1:
                    img = img[:, :, 0]
                im = Image.fromarray(img)
                fname = os.path.join(args.out_dir, f"{n_done:05d}.png")
                im.save(fname)
                n_done += 1

    print(f"Generated {n_done} images in '{args.out_dir}'")

if __name__ == "__main__":
    main()
