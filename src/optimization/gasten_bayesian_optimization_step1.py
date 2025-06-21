import os
import shutil
import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario
import wandb
from dotenv import load_dotenv

from src.utils.config     import read_config
from src.utils            import (
    MetricsLogger, group_images, load_z, setup_reprod,
    create_checkpoint_path, seed_worker
)
from src.utils.checkpoint import checkpoint_gan
from src.gan              import construct_gan, construct_loss
from src.gan.update_g     import UpdateGeneratorGAN
from src.gan.train        import train_disc, train_gen, evaluate
from src.metrics          import fid
from src.data_loaders     import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", dest="config_path", required=True,
                   help="YAML configuration file")
    return p.parse_args()


def init_distributed() -> int:
    """Initialize torch.distributed if launched via torchrun."""
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank


def is_primary() -> bool:
    return (not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0)


def main() -> None:
    load_dotenv()
    args       = parse_args()
    cfg        = read_config(args.config_path)
    local_rank = init_distributed()
    device     = torch.device(cfg["device"])

    # ───── Dataset ───────────────────────────────────────────────────────────
    ds_cfg   = cfg["dataset"]
    pos, neg = ds_cfg["binary"]["pos"], ds_cfg["binary"]["neg"]
    dataset, _, img_size = load_dataset(
        ds_cfg["name"], cfg["data-dir"], pos, neg
    )
    cfg["model"]["image-size"] = list(img_size)

    sampler = (DistributedSampler(dataset, shuffle=True)
               if dist.is_available() and dist.is_initialized()
               else None)

    dl = DataLoader(
        dataset,
        batch_size=cfg["train"]["step-1"]["batch-size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["num-workers"],
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # ───── FID setup ────────────────────────────────────────────────────────
    test_noise, _ = load_z(cfg["test-noise"])
    mu, sigma     = fid.load_statistics_from_path(cfg["fid-stats-path"])
    fm_fn, dims   = fid.get_inception_feature_map_fn(device)
    orig_fid      = fid.FID(fm_fn, dims, test_noise.size(0),
                            mu, sigma, device=device)
    fid_metrics   = {"fid": orig_fid}

    fixed_noise = torch.randn(
        cfg["fixed-noise"], cfg["model"]["z_dim"], device=device
    )

    # ───── WandB + checkpoint setup ────────────────────────────────────────
    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(cfg, run_id)

    if is_primary():
        wandb.init(
            project=f"{cfg['project']}-{pos}v{neg}",
            group=cfg["name"],
            entity=os.environ["ENTITY"],
            job_type="step-1",
            name=f"{run_id}-step1",
            config={
                "id": run_id,
                "gan": cfg["model"],
                "train": cfg["train"]["step-1"],
            },
        )

    train_metrics = MetricsLogger(prefix="train")
    eval_metrics  = MetricsLogger(prefix="eval")
    train_metrics.add("G_loss", iteration_metric=True)
    train_metrics.add("D_loss", iteration_metric=True)
    # ─── register the missing W_distance term ─────────────────────────────
    train_metrics.add("W_distance", iteration_metric=True)

    best_fid = float("inf")

    # ╭─────────────────────────────────────────────────────────────────────╮
    # │                      SMAC target function                         │
    # ╰─────────────────────────────────────────────────────────────────────╯
    def objective(params: Configuration, seed: int) -> float:
        nonlocal best_fid
        setup_reprod(seed)

        # update architecture hyper-params
        cfg["model"]["architecture"]["g_num_blocks"] = params["n_blocks"]
        cfg["model"]["architecture"]["d_num_blocks"] = params["n_blocks"]

        # build GAN + losses
        G, D      = construct_gan(cfg["model"], img_size, device)
        g_crit, d_crit = construct_loss(cfg["model"]["loss"], D)
        g_updater = UpdateGeneratorGAN(g_crit)

        # DataParallel or DDP
        if dist.is_available() and dist.is_initialized():
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            G = DDP(G.to(device), device_ids=[0])
            D = DDP(D.to(device), device_ids=[0])
        else:
            G, D = G.to(device), D.to(device)

        g_opt = Adam(G.parameters(), lr=params["g_lr"],
                     betas=(params["g_beta1"], params["g_beta2"]))
        d_opt = Adam(D.parameters(), lr=params["d_lr"],
                     betas=(params["d_beta1"], params["d_beta2"]))

        # ─── track generator & discriminator terms ─────────────────────────
        for term in g_updater.get_loss_terms():
            train_metrics.add(term, iteration_metric=True)
        for term in d_crit.get_loss_terms():
            train_metrics.add(term, iteration_metric=True)
        # (W_distance was missing in d_crit.get_loss_terms, so we added above)

        # ─── attach FID & images to eval metrics ───────────────────────────
        for k in fid_metrics:
            eval_metrics.add(k)
        eval_metrics.add_media_metric("samples")

        # ─── training loop ────────────────────────────────────────────────
        epochs     = cfg["train"]["step-1"].get("epochs", 30)
        disc_iters = cfg["train"]["step-1"]["disc-iters"]
        steps_per_epoch = (len(dl) // disc_iters) * disc_iters

        for epoch in range(1, epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)

            data_iter = iter(dl)
            g_steps = 0

            for i in range(1, steps_per_epoch + 1):
                real, _ = next(data_iter)
                real = real.to(device)

                d_loss, _ = train_disc(
                    G, D, d_opt, d_crit,
                    real, dl.batch_size,
                    train_metrics, device
                )

                if i % disc_iters == 0:
                    g_steps += 1
                    g_loss, _ = train_gen(
                        g_updater, G, D, g_opt,
                        dl.batch_size, train_metrics, device
                    )
                    if is_primary() and g_steps % cfg.get("log-every", 50) == 0:
                        print(f"[{epoch}/{epochs}]  "
                              f"G_loss={g_loss:.4f}  D_loss={d_loss:.4f}")

            # ─── evaluation on primary only ───────────────────────────────
            if is_primary():
                G.eval()
                with torch.no_grad():
                    fake = torch.cat([
                        G(fixed_noise[i:i+64]).cpu()
                        for i in range(0, fixed_noise.size(0), 64)
                    ])
                G.train()

                eval_metrics.log_image(
                    "samples",
                    group_images(fake, classifier=None, device=device)
                )
                train_metrics.finalize_epoch()
                evaluate(G, fid_metrics, eval_metrics,
                         dl.batch_size, test_noise, device, None)
                eval_metrics.finalize_epoch()

        # ─── checkpoint & prune ──────────────────────────────────────────
        run_folder = os.path.join(cp_dir, str(params.config_id))
        if is_primary():
            checkpoint_gan(
                G, D, g_opt, d_opt, None,
                {"train": train_metrics.stats,
                 "eval":  eval_metrics.stats},
                cfg, output_dir=run_folder
            )

        final_val = eval_metrics.stats["fid"][-1]
        if final_val < best_fid:
            best_fid = final_val
        else:
            if is_primary():
                shutil.rmtree(run_folder, ignore_errors=True)

        # broadcast to all ranks
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor([final_val], device=device)
            dist.broadcast(tensor, src=0)
            final_val = float(tensor.item())

        return final_val

    # ╭─────────────────────────────────────────────────────────────────────╮
    # │                   SMAC hyper-param space & run                     │
    # ╰─────────────────────────────────────────────────────────────────────╯
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        Float("g_lr",     (1e-4, 5e-4),   default=2e-4, log=True),
        Float("d_lr",     (1e-4, 5e-4),   default=2e-4, log=True),
        Float("g_beta1",  (0.0, 0.9),     default=0.5),
        Float("d_beta1",  (0.0, 0.9),     default=0.5),
        Float("g_beta2",  (0.9, 0.9999),  default=0.999),
        Float("d_beta2",  (0.9, 0.9999),  default=0.999),
        Integer("n_blocks", (1, 4),      default=3),
    ])

    scenario = Scenario(
        cs,
        deterministic=True,
        n_trials=cfg["train"]["step-1"].get("hpo-trials", 50),
        walltime_limit=cfg["train"]["step-1"].get("hpo-walltime", 4 * 3600),
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True
    )
    incumbent = smac.optimize()

    if is_primary():
        best_cfg = dict(incumbent)
        out_txt  = os.path.join(
            os.environ["FILESDIR"],
            f"step-1-best-config-{ds_cfg['name']}-{pos}v{neg}.txt"
        )
        with open(out_txt, "w") as f:
            f.write(os.path.join(cp_dir, str(incumbent.config_id)))

        print(f"Saved best config → {out_txt}")
        print("Best hyper-parameters:", best_cfg)
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()