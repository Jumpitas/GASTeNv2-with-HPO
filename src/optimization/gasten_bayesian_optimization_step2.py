import os, glob, json, argparse, numpy as np
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import wandb

from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, Float

from src.metrics.fid.fid_score import load_statistics_from_path
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
    load_z, set_seed, gen_seed, create_checkpoint_path,
    linear_warmup_cosine_decay,
)
from src.utils.config import read_config
from src.utils.checkpoint import (
    construct_gan_from_checkpoint,
    construct_classifier_from_checkpoint,
)
from src.gan import construct_loss


def construct_optimizers(opt_cfg, G, D):
    # generator: warmup+cosine LR scheduler
    g_optim = torch.optim.AdamW(
        G.parameters(), lr=opt_cfg["lr"], betas=(opt_cfg["beta1"], opt_cfg["beta2"]), weight_decay=1e-4
    )
    d_optim = torch.optim.AdamW(
        D.parameters(), lr=opt_cfg["lr"], betas=(opt_cfg["beta1"], opt_cfg["beta2"]), weight_decay=1e-4
    )
    return g_optim, d_optim


def train_modified_gan(
    config, dataset, cp_dir, gan_ckpt, test_noise,
    fid_metrics, c_out_hist,
    C, C_name, C_params, C_stats, C_args,
    weight, fixed_noise, num_classes, device, seed, run_id
):
    # pick updater
    if "gaussian" in weight:
        a, v = weight["gaussian"]["alpha"], weight["gaussian"]["var"]
        Updater = UpdateGeneratorGASTEN_gaussian
        tag = f"gaussα{a:.2f}_σ{v:.3f}"
    else:
        a, v = weight["gaussian-v2"]["alpha"], weight["gaussian-v2"]["var"]
        Updater = UpdateGeneratorGASTEN_gaussianV2
        tag = f"gaussV2α{a:.2f}_σ{v:.3f}"

    run_name = f"{C_name}_{tag}"
    out_dir  = os.path.join(cp_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # load step-1 GAN
    G, D, _, _ = construct_gan_from_checkpoint(gan_ckpt, device="cpu")

    if torch.cuda.device_count()>1:
        G = nn.DataParallel(G); D = nn.DataParallel(D)
        if hasattr(G.module, "z_dim"):
            G.z_dim = G.module.z_dim

    G, D = G.to(device), D.to(device)

    # losses + optimizers
    g_crit, d_crit = construct_loss(config["model"]["loss"], D)
    g_opt, d_opt   = construct_optimizers(config["optimizer"], G, D)
    g_updater      = Updater(g_crit, C, alpha=a, var=v)

    # LR-schedule: linear warmup 5% steps, then cosine decay
    total_steps = config["train"]["step-2"]["epochs"] * len(dataset) // config["train"]["step-2"]["batch-size"]
    scheduler_g = linear_warmup_cosine_decay(g_opt, total_steps, warmup_frac=0.05)
    scheduler_d = linear_warmup_cosine_decay(d_opt, total_steps, warmup_frac=0.05)

    # AMP scaler
    scaler = GradScaler()

    # EMA of G
    ema_G = G.__class__(**{**config["model"]["architecture"], "img_size":config["model"]["image-size"], "z_dim":config["model"]["z_dim"]})
    ema_G.load_state_dict(G.state_dict())
    ema_decay = 0.999

    # training hyperparams
    tcfg      = config["train"]["step-2"]
    bs, ne    = tcfg["batch-size"], tcfg["epochs"]
    nd, chk   = tcfg["disc-iters"], tcfg["checkpoint-every"]
    early_cfg = tcfg.get("early-stop", {})
    early_stop = (early_cfg.get("metric","fid_conf"), early_cfg.get("patience",10))
    # we'll stop on min( FID + 0.001*Conf )

    set_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    wandb.init(
        project=config["project"], group=config["name"], entity=os.getenv("ENTITY"),
        job_type="step-2", name=f"{run_id}-{run_name}",
        config={"seed":seed,"weight":weight,"train":tcfg,"classifier":C_name}
    )

    _, _, _, metrics = train(
        config, dataset, device, ne, bs,
        G, g_opt, g_updater,
        D, d_opt, d_crit,
        test_noise, fid_metrics,
        nd,
        early_stop=early_stop,
        checkpoint_dir=out_dir,
        fixed_noise=fixed_noise,
        c_out_hist=c_out_hist,
        checkpoint_every=chk,
        classifier=C,
        amp_scaler=scaler,
        schedulers=(scheduler_g, scheduler_d),
        ema_model=ema_G, ema_decay=ema_decay,
    )

    wandb.finish()
    return metrics


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = read_config(args.config)
    ds = cfg["dataset"]
    patt = f"results/step-1-best-config-{ds['name']}-{ds['binary']['pos']}v{ds['binary']['neg']}.txt"
    ckpts = glob.glob(patt)
    assert ckpts, f"No step1 ckpt for {patt}"
    gan_ckpt = open(ckpts[0]).read().strip()

    device = torch.device(cfg["device"])
    dataset, num_classes, _ = load_dataset(ds["name"], cfg["data-dir"], ds["binary"]["pos"], ds["binary"]["neg"])
    fixed_noise = torch.randn(cfg["fixed-noise"], cfg["model"]["z_dim"], device=device)
    test_noise, _ = load_z(cfg["test-noise"])

    mu, sigma     = load_statistics_from_path(cfg["fid-stats-path"])
    fm_fn, dims   = get_inception_feature_map_fn(device)
    original_fid  = FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    clf_path = cfg["train"]["step-2"]["classifier"][0]
    C, Cp, Cs, Ca = construct_classifier_from_checkpoint(clf_path, device=device)
    C.eval()

    fid_metrics = {
        "fid": original_fid,
        "conf_dist": LossSecondTerm(C),
        "hubris": Hubris(C, test_noise.size(0)),
    }
    c_out_hist = None if args.no_plots else OutputsHistogram(C, test_noise.size(0))

    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(cfg, run_id)

    def objective(hp_cfg, seed):
        return train_modified_gan(
            cfg, dataset, cp_dir, gan_ckpt, test_noise,
            fid_metrics, c_out_hist,
            C, os.path.basename(clf_path), Cp, Cs, Ca,
            {"gaussian": {"alpha":hp_cfg["alpha"], "var":hp_cfg["var"]}},
            fixed_noise, num_classes, device, seed, wandb.util.generate_id()
        ).stats["fid_conf"][-1]

    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        Float("alpha", (0.0,5.0), default=1.0),
        Float("var",   (1e-4,1.0), default=0.01),
    ])
    scenario = Scenario(cs, deterministic=True,
                        n_trials=cfg["train"]["step-2"].get("hpo-trials",50),
                        walltime_limit=cfg["train"]["step-2"].get("hpo-walltime",10000))
    smac = HyperparameterOptimizationFacade(scenario, objective, overwrite=True)
    best = smac.optimize()
    best_cfg = best.get_dictionary()

    with open(os.path.join(cp_dir, f"step2-best-gauss-{ds['binary']['pos']}v{ds['binary']['neg']}.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)

    # final retrain with best
    final = train_modified_gan(
        cfg, dataset, cp_dir, gan_ckpt, test_noise,
        fid_metrics, c_out_hist,
        C, os.path.basename(clf_path), Cp, Cs, Ca,
        {"gaussian": best_cfg},
        fixed_noise, num_classes, device,
        gen_seed(), wandb.util.generate_id()
    )
    # summary
    fid_list = final.stats["fid"]; cd_list = final.stats["conf_dist"]
    best_epoch = int(np.argmin(np.array(fid_list)+0.001*np.array(cd_list)))+1
    with open(os.path.join(cp_dir, "step2-summary.txt"), "w") as f:
        f.write(f"best α,σ² = {best_cfg}\nFID@best={fid_list[best_epoch-1]:.3f}\n")
    print("Step-2 done.")

if __name__=="__main__":
    main()
