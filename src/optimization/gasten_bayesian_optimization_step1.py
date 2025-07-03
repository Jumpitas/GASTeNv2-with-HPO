from __future__ import annotations
import argparse, math, os, shutil, torch
from dotenv import load_dotenv

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac import Scenario, HyperparameterOptimizationFacade
from torch.optim import Adam
import wandb

from src.utils.config import read_config
from src.data_loaders import load_dataset
from src.metrics import fid
from src.utils import (
    MetricsLogger, group_images, load_z, seed_worker,
    setup_reprod, create_checkpoint_path
)
from src.gan import construct_gan, construct_loss
from src.gan.update_g import UpdateGeneratorGAN
from src.gan.train import train_disc, train_gen, evaluate
from src.utils.checkpoint import checkpoint_gan


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML experiment config")
    p.add_argument("--trials", type=int, default=40,
                   help="SMAC budget (default 40)")
    p.add_argument("--walltime", type=int, default=20_000,
                   help="SMAC wall-time limit (s)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def main() -> None:
    load_dotenv()
    args   = parse_args()
    cfg    = read_config(args.config)
    device = torch.device(cfg["device"])

    # ------------------------------------------------------------- dataset ---
    ds_cfg = cfg["dataset"]
    pos_cls, neg_cls = ds_cfg["binary"]["pos"], ds_cfg["binary"]["neg"]
    dataset_name = ds_cfg["name"]

    dataset, _, img_size = load_dataset(
        dataset_name, cfg["data-dir"], pos_cls, neg_cls
    )
    cfg["model"]["image-size"] = list(img_size)

    batch_size   = cfg["train"]["step-1"]["batch-size"]
    n_disc_iters = cfg["train"]["step-1"]["disc-iters"]

    # ------------------------------------------------------------- FID setup -
    fm, dims      = fid.get_inception_feature_map_fn(device)
    mu, sigma     = fid.load_statistics_from_path(cfg["fid-stats-path"])
    test_noise, _ = load_z(cfg["test-noise"])
    test_noise    = torch.randn(10_000, cfg["model"]["z_dim"], device=device)
    fid_metric    = fid.FID(fm, dims, test_noise.size(0), mu, sigma, device)
    fid_metrics   = {"fid": fid_metric}

    fixed_noise = torch.randn(
        cfg["fixed-noise"], cfg["model"]["z_dim"], device=device
    )

    dl = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=cfg["num-workers"], worker_init_fn=seed_worker
    )

    # -------------------------------------------------------- bookkeeping ----
    cfg["project"] += f"-{pos_cls}v{neg_cls}"
    run_id  = wandb.util.generate_id()
    cp_root = create_checkpoint_path(cfg, run_id)
    best_fid = float("inf")

    wandb.init(
        project=cfg["project"],
        name=f"{run_id}-step1",
        group=cfg["name"],
        entity=os.environ["ENTITY"],
        job_type="step-1",
        config={"id": run_id,
                "gan":   cfg["model"],
                "train": cfg["train"]["step-1"]}
    )

    # ---------------------------------------------------------------- target -
    def objective(params: Configuration, seed: int) -> float:
        nonlocal best_fid
        setup_reprod(seed)

        # ------------ hyper-param injection -------------------------------
        arch = cfg["model"]["architecture"]
        arch["g_num_blocks"] = arch["d_num_blocks"] = params["n_blocks"]

        # ------------ build  +  optimisers -------------------------------
        G, D = construct_gan(cfg["model"], img_size, device)
        g_crit, d_crit = construct_loss(cfg["model"]["loss"], D)

        if torch.cuda.device_count() > 1:
            G, D = torch.nn.DataParallel(G), torch.nn.DataParallel(D)
            G.z_dim = G.module.z_dim

        G, D = G.to(device), D.to(device)
        g_up  = UpdateGeneratorGAN(g_crit)

        g_opt = Adam(G.parameters(), lr=params["g_lr"],
                     betas=(params["g_beta1"], params["g_beta2"]))
        d_opt = Adam(D.parameters(), lr=params["d_lr"],
                     betas=(params["d_beta1"], params["d_beta2"]))

        # ------------ loggers --------------------------------------------
        tr_log, ev_log = MetricsLogger("train"), MetricsLogger("eval")
        tr_log.add("G_loss", True); tr_log.add("D_loss", True)
        for t in g_up.get_loss_terms() + d_crit.get_loss_terms():
            tr_log.add(t, True)
        for m in fid_metrics: ev_log.add(m)
        ev_log.add_media_metric("samples")

        # ------------ training loop --------------------------------------
        epochs          = cfg["train"]["step-1"].get("epochs", 30)
        iters_per_epoch = (len(dl) // n_disc_iters) * n_disc_iters

        for ep in range(1, epochs + 1):
            it_dl = iter(dl)
            for i in range(1, iters_per_epoch + 1):
                real, _ = next(it_dl)
                train_disc(G, D, d_opt, d_crit,
                           real.to(device), batch_size, tr_log, device)
                if i % n_disc_iters == 0:
                    train_gen(g_up, G, D, g_opt,
                              batch_size, tr_log, device)

            # ---- evaluation -----------------------------------------
            with torch.no_grad():
                G.eval(); fake = G(fixed_noise).cpu(); G.train()
            ev_log.log_image("samples",
                             group_images(fake, None, device))

            tr_log.finalize_epoch()

            evaluate(
                G, fid_metrics, ev_log, batch_size,
                test_noise, device, c_out_hist=None,
                rgb_repeat=True         # <-- only name changed
            )
            ev_log.finalize_epoch()

        # ------------ checkpoint & prune ------------------------------
        run_dir = os.path.join(cp_root, str(params.config_id))
        os.makedirs(run_dir, exist_ok=True)
        checkpoint_gan(
            G, D, g_opt, d_opt, None,
            {"eval": ev_log.stats, "train": tr_log.stats},
            cfg, output_dir=run_dir
        )

        final_fid = ev_log.stats["fid"][-1]
        best_fid  = min(best_fid, final_fid)
        if final_fid > best_fid:
            shutil.rmtree(run_dir, ignore_errors=True)

        return final_fid

    # -------------------------------------------------- SMAC search space ----
    hp_space = ConfigurationSpace()
    hp_space.add_hyperparameters([
        Float("g_lr",  (1e-4, 5e-4), default=2e-4, log=True),
        Float("d_lr",  (1e-4, 5e-4), default=2e-4, log=True),
        Float("g_beta1", (0.0, 0.9), default=0.5),
        Float("d_beta1", (0.0, 0.9), default=0.5),
        Float("g_beta2", (0.9, 0.9999), default=0.999),
        Float("d_beta2", (0.9, 0.9999), default=0.999),
        Integer("n_blocks",
                (4, 5) if dataset_name in {"stl10", "chest-xray"}
                else (3, 4),
                default=4),
    ])

    scenario = Scenario(
        hp_space, deterministic=True,
        n_trials=args.trials, walltime_limit=args.walltime
    )
    smac      = HyperparameterOptimizationFacade(scenario, objective,
                                                 overwrite=True)
    incumbent = smac.optimize()

    # ------------------------------------------------------ save winner ------
    best_dir = os.path.join(cp_root, str(incumbent.config_id))
    out_path = os.path.join(
        os.environ["FILESDIR"],
        f"step1-best-{dataset_name}-{pos_cls}v{neg_cls}.txt"
    )
    with open(out_path, "w") as f:
        f.write(best_dir)

    print("Best config:", dict(incumbent))
    print("Saved checkpoint dir →", best_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
