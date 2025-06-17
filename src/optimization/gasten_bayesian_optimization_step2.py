import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import torch
import wandb

from src.metrics.fid.fid_score import (
    load_statistics_from_path,
    calculate_activation_statistics_dataloader,
)
from src.metrics.fid.inception import get_inception_feature_map_fn
from src.metrics.fid import FID

from src.metrics import LossSecondTerm, Hubris
from src.data_loaders import load_dataset
from src.gan.train import train
from src.gan.update_g import (
    UpdateGeneratorGASTEN_gaussian,
    UpdateGeneratorGASTEN_gaussianV2,
)
from src.metrics.c_output_hist import OutputsHistogram
from src.utils import (
    load_z,
    set_seed,
    setup_reprod,
    create_checkpoint_path,
    gen_seed,
    seed_worker,
)
from src.utils.plot import plot_metrics
from src.utils.config import read_config
from src.utils.checkpoint import (
    construct_gan_from_checkpoint,
    construct_classifier_from_checkpoint,
)
from src.gan import construct_loss  # ← STEP 1: bring in construct_loss

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True,
                        help="Path to YAML config")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip final plotting")
    return parser.parse_args()

def construct_optimizers(opt_cfg, G, D):
    g_optim = torch.optim.Adam(G.parameters(),
                               lr=opt_cfg["lr"],
                               betas=(opt_cfg["beta1"], opt_cfg["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(),
                               lr=opt_cfg["lr"],
                               betas=(opt_cfg["beta1"], opt_cfg["beta2"]))
    return g_optim, d_optim

def train_modified_gan(
    config, dataset, cp_dir, gan_ckpt, test_noise,
    fid_metrics, c_out_hist,
    C, C_name, C_params, C_stats, C_args,
    weight, fixed_noise, num_classes, device, seed, run_id
):
    # unpack gaussian vs gaussian-v2
    if "gaussian" in weight:
        a, v = weight["gaussian"]["alpha"], weight["gaussian"]["var"]
        updater_cls = UpdateGeneratorGASTEN_gaussian
        tag = f"gauss_α{a:.3f}_σ{v:.4f}"
    else:
        a, v = weight["gaussian-v2"]["alpha"], weight["gaussian-v2"]["var"]
        updater_cls = UpdateGeneratorGASTEN_gaussianV2
        tag = f"gaussV2_α{a:.3f}_σ{v:.4f}"

    run_name = f"{C_name}_{tag}"
    out_dir  = os.path.join(cp_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)  # ensures directory exists

    # load pretrained GAN from step‑1
    G, D, _, _ = construct_gan_from_checkpoint(gan_ckpt, device=device)
    g_crit, d_crit = construct_loss(config["model"]["loss"], D)
    g_opt, d_opt  = construct_optimizers(config["optimizer"], G, D)
    g_updater     = updater_cls(g_crit, C, alpha=a, var=v)

    # training hyperparams
    tcfg = config["train"]["step-2"]
    bs, nepochs = tcfg["batch-size"], tcfg["epochs"]
    nd, chk_every = tcfg["disc-iters"], tcfg["checkpoint-every"]
    early_crit = tcfg.get("early-stop", {}).get("criteria", None)
    early_stop = ("conf_dist", early_crit) if early_crit is not None else ("conf_dist", None)

    # reproducibility
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # wandb logging
    wandb.init(
        project=config["project"],
        group=config["name"],
        entity=os.environ["ENTITY"],
        job_type="step-2",
        name=f"{run_id}-{run_name}",
        config={
            "seed": seed,
            "weight": weight,
            "train": tcfg,
            "classifier": C_name,
            "classifier_loss": C_stats.get("test_loss", 0.0),
        },
    )

    # actual train loop
    _, _, _, metrics = train(
        config, dataset, device, nepochs, bs,
        G, g_opt, g_updater,
        D, d_opt, d_crit,
        test_noise, fid_metrics,
        nd,
        early_stop=early_stop,
        checkpoint_dir=out_dir,
        fixed_noise=fixed_noise,
        c_out_hist=c_out_hist,
        checkpoint_every=chk_every,
        classifier=C
    )

    wandb.finish()
    return metrics

def main():
    load_dotenv()
    args   = parse_args()
    config = read_config(args.config_path)
    print("Loaded config:", args.config_path)

    # find step‑1 best GAN checkpoint
    ds = config["dataset"]
    patt = f"results/step-1-best-config-{ds['name']}-{ds['binary']['pos']}v{ds['binary']['neg']}.txt"
    files = glob.glob(patt)
    if not files:
        raise FileNotFoundError(f"No step‑1 config matching {patt}")
    gan_ckpt = open(files[0]).read().strip()
    print("Using step‑1 GAN checkpoint:", gan_ckpt)

    # data + device
    device = torch.device(config["device"])
    dataset, num_classes, img_size = load_dataset(
        ds["name"], config["data-dir"], ds["binary"]["pos"], ds["binary"]["neg"]
    )

    # fixed & test noise
    if isinstance(config["fixed-noise"], str):
        fixed_noise = torch.Tensor(np.load(config["fixed-noise"])).to(device)
    else:
        fixed_noise = torch.randn(config["fixed-noise"], config["model"]["z_dim"], device=device)
    test_noise, _ = load_z(config["test-noise"])

    # FID setup
    mu, sigma      = load_statistics_from_path(config["fid-stats-path"])
    fm_fn, dims    = get_inception_feature_map_fn(device)
    original_fid   = FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    # classifier
    clf_path = config["train"]["step-2"]["classifier"][0]
    C, C_p, C_s, C_a = construct_classifier_from_checkpoint(clf_path, device=device)
    C.eval()

    # metrics dictionary
    fid_metrics = {
        "fid":       original_fid,
        "conf_dist": LossSecondTerm(C),
        "hubris":    Hubris(C, test_noise.size(0)),
    }
    c_out_hist = None if args.no_plots else OutputsHistogram(C, test_noise.size(0))

    # experiment folder
    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)

    # composite weights (you can tweak these)
    w_fid, w_cd = 1.0, 0.001

    # HPO objective: composite of FID + tiny weight × conf_dist
    def step2_obj(cfg: Configuration, seed: int) -> float:
        metrics = train_modified_gan(
            config, dataset, cp_dir, gan_ckpt, test_noise,
            fid_metrics, c_out_hist,
            C, os.path.basename(clf_path), C_p, C_s, C_a,
            {"gaussian": {"alpha": cfg["alpha"], "var": cfg["var"]}},
            fixed_noise, num_classes, device, seed,
            wandb.util.generate_id()
        )
        # guard empty
        if not metrics.stats["fid"] or not metrics.stats["conf_dist"]:
            return float("inf")
        fid_val = metrics.stats["fid"][-1]
        cd_val  = metrics.stats["conf_dist"][-1]
        return w_fid * fid_val + w_cd * cd_val

    # SMAC setup
    cs       = ConfigurationSpace()
    cs.add_hyperparameters([
        Float("alpha", (0.0, 5.0), default=1.0),
        Float("var",   (1e-4, 1.0), default=0.01),
    ])
    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=config["train"]["step-2"].get("hpo-trials", 50),
        walltime_limit=config["train"]["step-2"].get("hpo-walltime", 36000),
    )
    smac = HyperparameterOptimizationFacade(scenario, step2_obj, overwrite=True)

    print("Starting SMAC HPO step‑2 …")
    incumbent = smac.optimize()
    best_cfg   = incumbent.get_dictionary()

    # save JSON of best hyperparams
    out_json = os.path.join(
        cp_dir,
        f"step-2-best-gauss-{ds['binary']['pos']}v{ds['binary']['neg']}.json"
    )
    with open(out_json, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print("→ Saved best hyperparams to", out_json)

    # final evaluation + summary TXT
    final_metrics = train_modified_gan(
        config, dataset, cp_dir, gan_ckpt, test_noise,
        fid_metrics, c_out_hist,
        C, os.path.basename(clf_path), C_p, C_s, C_a,
        {"gaussian": best_cfg},
        fixed_noise, num_classes, device,
        gen_seed(), wandb.util.generate_id()
    )

    fid_list = final_metrics.stats["fid"]
    best_epoch = int(np.argmin(fid_list)) + 1

    last_fid = final_metrics.stats["fid"][-1]
    last_cd  = final_metrics.stats["conf_dist"][-1]

    summary_txt = os.path.join(
        cp_dir,
        f"step-2-best-gauss-{ds['binary']['pos']}v{ds['binary']['neg']}-summary.txt"
    )
    with open(summary_txt, "w") as f:
        f.write("Best Gaussian parameters:\n")
        f.write(json.dumps(best_cfg, indent=2) + "\n\n")
        f.write("Performance metrics:\n")
        f.write(f"  FID       = {last_fid:.4f}\n")
        f.write(f"  Conf_dist = {last_cd:.4f}\n")
        f.write(f"  Best    Epoch      = {best_epoch}\n")
    print("→ Saved summary to", summary_txt)

    # optional plots
    if not args.no_plots:
        df = pd.DataFrame({
            "fid":       final_metrics.stats["fid"],
            "conf_dist": final_metrics.stats["conf_dist"],
            "hubris":    final_metrics.stats["hubris"],
            "epoch":     list(range(1, len(final_metrics.stats["fid"]) + 1)),
        })
        plot_metrics(df, cp_dir, f"final-{ds['binary']['pos']}v{ds['binary']['neg']}")

if __name__ == "__main__":
    main()
