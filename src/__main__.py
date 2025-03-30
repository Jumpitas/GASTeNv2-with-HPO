import os
import argparse
import numpy as np
from dotenv import load_dotenv
import pandas as pd

import torch
import wandb

from src.metrics import fid, LossSecondTerm, Hubris
from src.datasets import load_dataset
from src.gan.train import train
from src.gan.update_g import (UpdateGeneratorGAN, UpdateGeneratorGASTEN_gaussian,
                              UpdateGeneratorGASTEN_gaussianV2)
from src.metrics.c_output_hist import OutputsHistogram
from src.utils import load_z, set_seed, setup_reprod, create_checkpoint_path, gen_seed, seed_worker
from src.utils.plot import plot_metrics
from src.utils.config import read_config
from src.utils.checkpoint import (construct_gan_from_checkpoint, construct_classifier_from_checkpoint,
                                  get_gan_path_at_epoch, load_gan_train_state)
from src.gan import construct_gan, construct_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Path to config file")
    return parser.parse_args()


def construct_optimizers(config, G, D):
    g_optim = torch.optim.Adam(G.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    d_optim = torch.optim.Adam(D.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    return g_optim, d_optim


def train_modified_gan(config, dataset, cp_dir, gan_path, test_noise,
                       fid_metrics, c_out_hist,
                       C, C_name, C_params, C_stats, C_args, weight, fixed_noise, num_classes, device, seed, run_id,
                       s1_epoch):
    print("Running experiment with classifier {} and weight {} ...".format(C_name, weight))

    # Ensure weight is a dictionary for gaussian loss
    if not (isinstance(weight, dict) and ("gaussian" in weight or "gaussian-v2" in weight)):
        raise ValueError(
            "For GASTeN v2 we require weight to be specified as a dictionary with 'gaussian' or 'gaussian-v2'.")

    if "gaussian" in weight:
        weight_txt = 'gauss_' + '_'.join([f'{key}_{value}' for key, value in weight['gaussian'].items()])
    elif "gaussian-v2" in weight:
        weight_txt = 'gauss_v2_' + '_'.join([f'{key}_{value}' for key, value in weight['gaussian-v2'].items()])

    # s1_epoch may be "best", "last", or a number as a string.
    if s1_epoch == "best":
        epoch = step_1_train_state['best_epoch']
    elif s1_epoch == "last":
        epoch = step_1_train_state['epoch']
    else:
        epoch = int(s1_epoch)

    run_name = '{}_{}_{}'.format(C_name, weight_txt, s1_epoch)
    gan_cp_dir = os.path.join(cp_dir, run_name)

    batch_size = config['train']['step-2']['batch-size']
    n_epochs = config['train']['step-2']['epochs']
    n_disc_iters = config['train']['step-2']['disc-iters']
    checkpoint_every = config['train']['step-2']['checkpoint-every']

    # Load GAN checkpoint for the given epoch.
    G, D, _, _ = construct_gan_from_checkpoint(gan_path, device=device)
    g_crit, d_crit = construct_loss(config["model"]["loss"], D)
    g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)

    if "gaussian" in weight:
        alpha = weight["gaussian"]["alpha"]
        var = weight["gaussian"]["var"]
        g_updater = UpdateGeneratorGASTEN_gaussian(g_crit, C, alpha=alpha, var=var)
    elif "gaussian-v2" in weight:
        alpha = weight["gaussian-v2"]["alpha"]
        var = weight["gaussian-v2"]["var"]
        g_updater = UpdateGeneratorGASTEN_gaussianV2(g_crit, C, alpha=alpha, var=var)
    else:
        raise NotImplementedError("Only Gaussian loss variants are supported.")

    early_stop_key = 'conf_dist'
    early_stop_crit = config['train']['step-2'].get('early-stop', {}).get('criteria', None)
    early_stop_crit_step_1 = config['train']['step-1'].get('early-stop', {}).get('criteria', None)
    early_stop = (early_stop_key, early_stop_crit) if early_stop_crit is not None else (early_stop_key, None)

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
                   'classifier_loss': C_stats['test_loss'],
                   'classifier': C_name,
                   'classifier_args': C_args,
                   'classifier_params': C_params,
                   'step1_epoch': s1_epoch
               })

    _, _, _, eval_metrics = train(
        config, dataset, device, n_epochs, batch_size,
        G, g_optim, g_updater,
        D, d_optim, d_crit,
        test_noise, fid_metrics,
        n_disc_iters,
        early_stop=early_stop,
        start_early_stop_when=('fid', early_stop_crit_step_1),
        checkpoint_dir=gan_cp_dir, fixed_noise=fixed_noise, c_out_hist=c_out_hist,
        checkpoint_every=checkpoint_every,
        classifier=C)

    wandb.finish()
    return eval_metrics


def compute_dataset_fid_stats(dset, get_feature_map_fn, dims, batch_size=64, device='cpu', num_workers=0):
    dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers,
                                             worker_init_fn=seed_worker)
    m, s = fid.calculate_activation_statistics_dataloader(dataloader, get_feature_map_fn, dims=dims, device=device)
    return m, s


def main():
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)
    print("Loaded experiment configuration from {}".format(args.config_path))

    run_seeds = config.get("step-1-seeds", [gen_seed() for _ in range(config["num-runs"])])
    step_2_seeds = config.get("step-2-seeds", [gen_seed() for _ in range(config["num-runs"])])

    device = torch.device(config["device"])
    print("Using device {}".format(device))

    # Load dataset for binary classification (e.g., MNIST 5 vs. 3)
    pos_class = config["dataset"].get("binary", {}).get("pos", None)
    neg_class = config["dataset"].get("binary", {}).get("neg", None)
    dataset, num_classes, img_size = load_dataset(config["dataset"]["name"], config["data-dir"], pos_class, neg_class)
    num_workers = config["num-workers"]
    print(" > Num workers", num_workers)

    if isinstance(config['fixed-noise'], str):
        arr = np.load(config['fixed-noise'])
        fixed_noise = torch.Tensor(arr).to(device)
    else:
        fixed_noise = torch.randn(config['fixed-noise'], config["model"]["z_dim"], device=device)

    test_noise, test_noise_conf = load_z(config['test-noise'])
    print("Loaded test noise from", config['test-noise'])
    print("\t", test_noise_conf)

    mu, sigma = fid.load_statistics_from_path(config['fid-stats-path'])
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(fm_fn, dims, test_noise.size(0), mu, sigma, device=device)

    # ---- Step 1: Baseline GAN Training ----
    num_runs = config["num-runs"]
    for i in range(num_runs):
        print("##\n# Starting run", i, "\n##")
        run_id = wandb.util.generate_id()
        cp_dir = create_checkpoint_path(config, run_id)
        with open(os.path.join(cp_dir, 'fixed_noise.npy'), 'wb') as f:
            np.save(f, fixed_noise.cpu().numpy())

        seed = run_seeds[i]
        setup_reprod(seed)
        config["model"]["image-size"] = img_size

        # Train baseline GAN (Step 1)
        G, D = construct_gan(config["model"], img_size, device)
        g_optim, d_optim = construct_optimizers(config["optimizer"], G, D)
        g_crit, d_crit = construct_loss(config["model"]["loss"], D)
        g_updater = UpdateGeneratorGAN(g_crit)
        print("Storing generated artifacts in {}".format(cp_dir))
        original_gan_cp_dir = os.path.join(cp_dir, 'step-1')

        if not isinstance(config['train']['step-1'], str):
            batch_size = config['train']['step-1']['batch-size']
            n_epochs = config['train']['step-1']['epochs']
            n_disc_iters = config['train']['step-1']['disc-iters']
            checkpoint_every = config['train']['step-1']['checkpoint-every']

            fid_metrics = {'fid': original_fid}
            early_stop_key = 'fid'
            early_stop_crit = config['train']['step-1'].get('early-stop', {}).get('criteria', None)
            early_stop = (early_stop_key, early_stop_crit) if early_stop_crit is not None else (early_stop_key, None)

            wandb.init(project=config["project"],
                       group=config["name"],
                       entity=os.environ['ENTITY'],
                       job_type='step-1',
                       name=f'{run_id}-step-1',
                       config={
                           'id': run_id,
                           'seed': seed,
                           'gan': config["model"],
                           'optim': config["optimizer"],
                           'train': config["train"]["step-1"],
                           'dataset': config["dataset"],
                           'num-workers': config["num-workers"],
                           'test-noise': test_noise_conf,
                       })

            step_1_train_state, _, _, _ = train(
                config, dataset, device, n_epochs, batch_size,
                G, g_optim, g_updater,
                D, d_optim, d_crit,
                test_noise, fid_metrics,
                n_disc_iters,
                early_stop=early_stop,
                checkpoint_dir=original_gan_cp_dir, fixed_noise=fixed_noise,
                checkpoint_every=checkpoint_every)
            wandb.finish()
        else:
            original_gan_cp_dir = config['train']['step-1']
            step_1_train_state = load_gan_train_state(original_gan_cp_dir)

        # ---- Step 2: Modified GAN Training (GASTeN v2) ----
        print(" > Start step 2 (GAN with modified loss)")
        # Use only the first checkpoint epoch from step 1
        step_1_epochs = config['train']['step-2'].get('step-1-epochs', ["best"])
        s1_epoch_str = step_1_epochs[0]
        if s1_epoch_str == "best":
            epoch = step_1_train_state['best_epoch']
        elif s1_epoch_str == "last":
            epoch = step_1_train_state['epoch']
        else:
            epoch = int(s1_epoch_str)

        gan_checkpoint_path = get_gan_path_at_epoch(original_gan_cp_dir, epoch=epoch)

        # Use only a single classifier: choose the first one.
        classifier_path = config['train']['step-2']['classifier'][0]
        C_name = os.path.splitext(os.path.basename(classifier_path))[0]
        C, C_params, C_stats, C_args = construct_classifier_from_checkpoint(classifier_path, device=device)
        C.to(device)
        C.eval()

        # Feature-map extraction: Pool spatial dimensions and flatten to a 1D prediction.
        def get_feature_map_fn(images, batch_idx, batch_size):
            output = C(images, output_feature_maps=True)
            feature_map = output[-2]  # shape: (batch, dims, h, w)
            pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
            return pooled.view(pooled.size(0), -1)

        dims = get_feature_map_fn(dataset.data[0:1].to(device), 0, 1).size(1)
        print(" > Computing statistics using original dataset")
        mu, sigma = compute_dataset_fid_stats(dataset, get_feature_map_fn, dims, device=device, num_workers=num_workers)
        print("   ... done")
        our_class_fid = fid.FID(get_feature_map_fn, dims, test_noise.size(0), mu, sigma, device=device)

        # Compute additional metrics using the classifier.
        conf_dist = LossSecondTerm(C)
        fid_metrics = {
            'fid': original_fid,
            'focd': our_class_fid,
            'conf_dist': conf_dist,
            'hubris': Hubris(C, test_noise.size(0)),
        }
        c_out_hist = OutputsHistogram(C, test_noise.size(0))

        weight = config['train']['step-2']['weight'][0]
        eval_metrics = train_modified_gan(config, dataset, cp_dir,
                                          gan_checkpoint_path,
                                          test_noise, fid_metrics, c_out_hist,
                                          C, C_name, C_params, C_stats, C_args,
                                          weight, fixed_noise, num_classes, device,
                                          step_2_seeds[i], run_id, epoch)

        step2_metrics = pd.DataFrame({
            'fid': eval_metrics.stats['fid'],
            'conf_dist': eval_metrics.stats['conf_dist'],
            'hubris': eval_metrics.stats['hubris'],
            's1_epochs': [epoch] * len(eval_metrics.stats['fid']),
            'weight': [str(weight)] * len(eval_metrics.stats['fid']),
            'classifier': [classifier_path] * len(eval_metrics.stats['fid']),
            'epoch': [i + 1 for i in range(len(eval_metrics.stats['fid']))]
        })
        plot_metrics(step2_metrics, cp_dir, f'{C_name}-{run_id}')

if __name__ == "__main__":
    main()
