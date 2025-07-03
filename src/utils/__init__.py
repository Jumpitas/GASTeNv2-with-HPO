import os
from datetime import datetime
import itertools
import random
import sys
import torch
import torch.nn.functional as F
import numpy as np
import math
import subprocess
import json
import torchvision.utils as vutils
from .metrics_logger import MetricsLogger


def create_checkpoint_path(config, run_id):
    path = os.path.join(
        config['out-dir'],
        config['project'],
        config['name'],
        datetime.now().strftime(f'%b%dT%H-%M_{run_id}')
    )
    os.makedirs(path, exist_ok=True)
    return path


def create_exp_path(config):
    path = os.path.join(config['out-dir'], config['name'])
    os.makedirs(path, exist_ok=True)
    return path


def gen_seed(max_val=10000):
    return np.random.randint(max_val)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_reprod(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    set_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_and_store_z(out_dir, n, dim, name=None, config=None):
    if name is None:
        name = f"z_{n}_{dim}"
    noise = torch.randn(n, dim).numpy()
    out_path = os.path.join(out_dir, name)
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, 'z.npy'), 'wb') as f:
        np.savez(f, z=noise)
    if config is not None:
        with open(os.path.join(out_path, 'z.json'), "w") as out_json:
            json.dump(config, out_json)
    return torch.Tensor(noise), out_path


def load_z(path):
    with np.load(os.path.join(path, 'z.npy')) as f:
        z = f['z'][:]
    with open(os.path.join(path, 'z.json')) as f:
        conf = json.load(f)
    return torch.Tensor(z), conf


def make_grid(images, nrow=None, total_images=None):
    if nrow is None:
        # try square
        nrow = math.isqrt(images.size(0))
        if nrow * nrow != images.size(0):
            nrow = 8
    else:
        if total_images is not None:
            total_images = math.ceil(total_images / nrow) * nrow
            pad = total_images - images.size(0)
            if pad > 0:
                blank = -torch.ones(
                    (pad, images.size(1), images.size(2), images.size(3)),
                    device=images.device
                )
                images = torch.cat([images, blank], dim=0)
    return vutils.make_grid(
        images, padding=2, normalize=True, nrow=int(nrow), value_range=(-1, 1)
    )


def group_images(images, classifier=None, device=None):
    if classifier is None:
        return make_grid(images)

    n_images = images.size(0)
    y = torch.zeros(n_images, device=device)

    # score in batches to avoid OOM
    for start in range(0, n_images, 100):
        stop = min(start + 100, n_images)
        batch = images[start:stop].to(device)
        out = classifier(batch)

        if out.ndim == 2 and out.size(1) == 2:
            probs = F.softmax(out, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(out.view(-1))

        y[start:stop] = probs

    y, idxs = torch.sort(y)
    images = images.to(device)[idxs]

    n_divs = 10
    step = 1.0 / n_divs
    groups = []
    lo = 0
    largest = 0

    for i in range(n_divs):
        up = (i + 1) * step
        mask = (y > up).nonzero(as_tuple=True)[0]
        hi = mask[0].item() if mask.numel() > 0 else n_images
        groups.append(images[lo:hi])
        largest = max(largest, hi - lo)
        lo = hi

    if largest == 0:
        return make_grid(images)

    nrow = 3
    C0, H0, W0 = images.size(1), images.size(2), images.size(3)
    dummy = torch.zeros(largest, C0, H0, W0, device=device)
    ref_grid = make_grid(dummy, nrow=nrow, total_images=largest)
    C_out, H_out, W_out = ref_grid.shape

    grids = []
    for g in groups:
        if g.numel() == 0:
            grids.append(torch.zeros((C_out, H_out, W_out), device=device))
        else:
            padded = make_grid(g, nrow=nrow, total_images=largest)
            grids.append(padded)

    return torch.cat(grids, dim=2)


def begin_classifier(iterator, clf_type, l_epochs, args):
    os.makedirs(args.out_dir, exist_ok=True)
    l_nf = [nf for nf in args.nf.split(",") if nf.isdigit()]
    print("nf candidates:", l_nf)
    print("epochs candidates:", l_epochs)

    class_pairs = list(iterator)
    print("class pairs:", class_pairs)

    for neg_class, pos_class in class_pairs:
        print(f"\n=== {pos_class} vs {neg_class} ===")
        for nf, epochs in itertools.product(l_nf, l_epochs):
            print(f"-- {clf_type} | nf={nf} | epochs={epochs}")
            cmd = list(map(str, [
                sys.executable, "-m", "src.classifier.train",
                "--device",        args.device,
                "--data-dir",      args.dataroot,
                "--out-dir",       args.out_dir,
                "--dataset",       args.dataset,
                "--pos",           pos_class,
                "--neg",           neg_class,
                "--classifier-type", clf_type,
                "--nf",            nf,
                "--epochs",        epochs,
                "--batch-size",    args.batch_size,
                "--lr",            args.lr,
                "--seed",          args.seed,
            ]))
            print("running:", " ".join(cmd))
            result = subprocess.run(cmd, cwd=os.getcwd())
            if result.returncode != 0:
                print(f"FAILED (code {result.returncode})")
            else:
                print("DONE — check", args.out_dir)
