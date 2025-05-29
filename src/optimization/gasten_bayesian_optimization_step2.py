import os
import glob
import json
import argparse
import numpy as np
from dotenv import load_dotenv
import pandas as pd

import torch
import wandb

from src.metrics import fid, LossSecondTerm, Hubris
from src.data_loaders import load_dataset
from src.gan.train import train
from src.gan.update_g import (UpdateGeneratorGAN, UpdateGeneratorGASTEN_gaussian,
                              UpdateGeneratorGASTEN_gaussianV2)
from src.metrics.c_output_hist import OutputsHistogram
from src.utils import load_z, set_seed, setup_reprod, create_checkpoint_path, gen_seed, seed_worker
from src.utils.plot import plot_metrics
from src.utils.config import read_config
from src.utils.checkpoint import (construct_gan_from_checkpoint, construct_classifier_from_checkpoint)
from src.gan import construct_gan, construct_loss

# SMAC imports for Bayesian
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Path to config file")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable creation of images for plots")
    return parser.parse_args()

def construct_optimizers(config, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    return g_optim, d_optim

def train_modified_gan(config, dataset, cp_dir, gan_path, test_noise,
                       fid_metrics, c_out_hist,
                       C, C_name, C_params, C_stats, C_args, weight, fixed_noise, num_classes, device, seed, run_id):
    print("Running experiment with classifier {} and weight {} ...".format(C_name, weight))

    if not (isinstance(weight, dict) and ("gaussian" in weight or "gaussian-v2" in weight)):
        raise ValueError("For GASTeN v2 we require a dict with 'gaussian' or 'gaussian-v2'.")

    if "gaussian" in weight:
        weight_txt = 'gauss_' + '_'.join([f'{k}_{v}' for k, v in weight['gaussian'].items()])
    else:
        weight_txt = 'gauss_v2_' + '_'.join([f'{k}_{v}' for k, v in weight['gaussian-v2'].items()])

    run_name = f"{C_name}_{weight_txt}"
    gan_cp_dir = os.path.join(cp_dir, run_name)

    batch_size     = config['train']['step-2']['batch-size']
    n_epochs       = config['train']['step-2']['epochs']
    n_disc_iters   = config['train']['step-2']['disc-iters']
    checkpoint_every = config['train']['step-2']['checkpoint-every']

    G, D, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    g_crit, d_crit = construct_loss(config["model"]["loss"], D)
    g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)

    if "gaussian" in weight:
        a, v = weight["gaussian"]["alpha"], weight["gaussian"]["var"]
        g_updater = UpdateGeneratorGASTEN_gaussian(g_crit, C, alpha=a, var=v)
    else:
        a, v = weight["gaussian-v2"]["alpha"], weight["gaussian-v2"]["var"]
        g_updater = UpdateGeneratorGASTEN_gaussianV2(g_crit, C, alpha=a, var=v)

    early_stop_key  = 'conf_dist'
    crit           = config['train']['step-2'].get('early-stop', {}).get('criteria', None)
    early_stop     = (early_stop_key, crit) if crit is not None else (early_stop_key, None)

    set_seed(seed)
    wandb.init(project=config["project"],
               group=config["name"],
               entity=os.environ['ENTITY'],
               job_type='step-2',
               name=f'{run_id}-{run_name}',
               config={
                   'id': run_id,
                   'seed': seed,
                   'weight': weight_txt,
                   'train': config["train"]["step-2"],
                   'classifier': C_name,
                   'classifier_loss': C_stats.get('test_loss', 0.0),
               })

    _, _, _, eval_metrics = train(
        config, dataset, device, n_epochs, batch_size,
        G, g_optim, g_updater,
        D, d_optim, d_crit,
        test_noise, fid_metrics,
        n_disc_iters,
        early_stop=early_stop,
        checkpoint_dir=gan_cp_dir,
        fixed_noise=fixed_noise,
        c_out_hist=c_out_hist,
        checkpoint_every=checkpoint_every,
        classifier=C
    )

    wandb.finish()
    return eval_metrics

def compute_dataset_fid_stats(dset, fn, dims, batch_size=64, device='cpu', num_workers=0):
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers,
                                         worker_init_fn=seed_worker)
    return fid.calculate_activation_statistics_dataloader(loader, fn, dims=dims, device=device)

def main():
    load_dotenv()
    args   = parse_args()
    config = read_config(args.config_path)
    print("Loaded config:", args.config_path)

    # find the step1 best‐config file (with any run_id suffix)
    pattern = f"results/step-1-best-config-bayesian-{config['dataset']['binary']['pos']}v{config['dataset']['binary']['neg']}-*.txt"
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No step‐1 best‐config file matching " + pattern)
    gan_checkpoint_dir = open(files[0]).read().strip()
    config['train']['step-1'] = gan_checkpoint_dir
    print("Using GAN checkpoint:", gan_checkpoint_dir)

    # seeds
    step_2_seeds = config.get("step-2-seeds", [gen_seed() for _ in range(config["num-runs"])])

    device = torch.device(config["device"])
    pos, neg = config["dataset"]["binary"]["pos"], config["dataset"]["binary"]["neg"]
    dataset, num_classes, img_size = load_dataset(config["dataset"]["name"], config["data-dir"], pos, neg)
    print("Dataset:", config["dataset"]["name"], f"{pos}v{neg}")

    fixed_noise = (
        torch.Tensor(np.load(config['fixed-noise']))[...,] .to(device)
        if isinstance(config['fixed-noise'], str)
        else torch.randn(config['fixed-noise'], config["model"]["z_dim"], device=device)
    )
    test_noise, _ = load_z(config['test-noise'])

    mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    # load classifier
    clf_path = config['train']['step-2']['classifier'][0]
    C, C_params, C_stats, C_args = construct_classifier_from_checkpoint(clf_path, device=device)
    C.eval()

    # prepare metrics and hist
    fid_metrics = {
        'fid': original_fid,
        'conf_dist': LossSecondTerm(C),
        'hubris': Hubris(C, test_noise.size(0))
    }
    c_out_hist = None if args.no_plots else OutputsHistogram(C, test_noise.size(0))

    # cp_dir & gan_path
    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)
    gan_path = gan_checkpoint_dir

    # objective for SMAC
    def step2_obj(cfg: Configuration, seed: int) -> float:
        weight = {"gaussian": {"alpha": cfg["alpha"], "var": cfg["var"]}}
        em = train_modified_gan(
            config, dataset, cp_dir, gan_path, test_noise,
            fid_metrics, c_out_hist,
            C, os.path.basename(clf_path), C_params, C_stats, C_args,
            weight, fixed_noise, num_classes, device, seed, wandb.util.generate_id()
        )
        return em.stats['fid'][-1]

    # Bayesian over alpha∈[0,5], var∈[1e-4,1]
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        Float("alpha", (0.0, 5.0), default=1.0),
        Float("var",   (1e-4, 1.0), default=0.01),
    ])
    scenario = Scenario(cs, deterministic=True, n_trials=50, walltime_limit=72000)
    smac = HyperparameterOptimizationFacade(scenario, step2_obj, overwrite=True)
    incumbent = smac.optimize()
    best = incumbent.get_dictionary()

    out_json = f"results/step-2-best-gauss-{pos}v{neg}.json"
    with open(out_json, "w") as f:
        json.dump(best, f, indent=2)
    print("Saved step‑2 best Gaussian params →", out_json)

    # (optional) final run with best values for plots
    if not args.no_plots:
        weight = {"gaussian": best}
        em = train_modified_gan(
            config, dataset, cp_dir, gan_path, test_noise,
            fid_metrics, c_out_hist,
            C, os.path.basename(clf_path), C_params, C_stats, C_args,
            weight, fixed_noise, num_classes, device, gen_seed(), wandb.util.generate_id()
        )
        df = pd.DataFrame({
            'fid': em.stats['fid'],
            'conf_dist': em.stats['conf_dist'],
            'hubris': em.stats['hubris'],
            'epoch': list(range(1, len(em.stats['fid'])+1))
        })
        plot_metrics(df, cp_dir, f"final-{pos}v{neg}")

if __name__ == "__main__":
    main()
