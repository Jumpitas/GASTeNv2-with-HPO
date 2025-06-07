import torch
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario
from torch.optim import Adam
import argparse
from dotenv import load_dotenv
import wandb
import os
import math
import shutil

from src.utils.config import read_config
from src.gan import construct_gan, construct_loss
from src.data_loaders import load_dataset
from src.gan.update_g import UpdateGeneratorGAN
from src.metrics import fid
from src.utils import MetricsLogger, group_images
from src.gan.train import train_disc, train_gen, loss_terms_to_str, evaluate
from src.utils.checkpoint import checkpoint_gan
from src.utils import load_z, setup_reprod, create_checkpoint_path, seed_worker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path",
                        required=True, help="Config file")
    parser.add_argument('--pos', dest='pos_class', default=9,
                        type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=4,
                        type=int, help='Negative class for binary classification')
    parser.add_argument('--dataset', dest='dataset',
                        default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    config = read_config(args.config_path)
    device = torch.device(config["device"])

    pos_class, neg_class = args.pos_class, args.neg_class
    dataset, num_classes, img_size = load_dataset(
        args.dataset, config["data-dir"], pos_class, neg_class)

    config['model']['image-size'] = list(img_size)
    n_disc_iters = config['train']['step-1']['disc-iters']
    test_noise, _ = load_z(config['test-noise'])
    batch_size = config['train']['step-1']['batch-size']

    fid_stats_path = (
        f"{os.environ['FILESDIR']}/data/fid-stats/"
        f"stats.inception.{args.dataset}.{pos_class}v{neg_class}.npz"
    )
    mu, sigma = fid.load_statistics_from_path(fid_stats_path)
    fm_fn, dims = fid.get_inception_feature_map_fn(device)
    original_fid = fid.FID(
        fm_fn, dims, test_noise.size(0), mu, sigma, device=device
    )
    fid_metrics = {'fid': original_fid}

    fixed_noise = torch.randn(
        config['fixed-noise'], config['model']['z_dim'], device=device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=config['num-workers'], worker_init_fn=seed_worker)

    config["project"] = f"{config['project']}-{pos_class}v{neg_class}"
    run_id = wandb.util.generate_id()
    cp_dir = create_checkpoint_path(config, run_id)

    # track best FID to prune inferior runs
    best_so_far = float('inf')

    # initialize wandb in normal mode
    wandb.init(
        project=config["project"],
        group=config["name"],
        entity=os.environ['ENTITY'],
        job_type='step-1',
        name=f'{run_id}-step-1',
        config={'id': run_id, 'gan': config['model'], 'train': config['train']['step-1']},
    )

    train_metrics = MetricsLogger(prefix='train')
    eval_metrics = MetricsLogger(prefix='eval')
    train_metrics.add('G_loss', iteration_metric=True)
    train_metrics.add('D_loss', iteration_metric=True)

    def train(params: Configuration, seed) -> float:
        nonlocal best_so_far
        setup_reprod(seed)
        config['model']["architecture"]['g_num_blocks'] = params['n_blocks']
        config['model']["architecture"]['d_num_blocks'] = params['n_blocks']

        G, D = construct_gan(config["model"], img_size, device)
        g_crit, d_crit = construct_loss(config["model"]["loss"], D)
        g_updater = UpdateGeneratorGAN(g_crit)

        g_opt = Adam(G.parameters(), lr=params['g_lr'], betas=(params['g_beta1'], params['g_beta2']))
        d_opt = Adam(D.parameters(), lr=params['d_lr'], betas=(params['d_beta1'], params['d_beta2']))

        for term in g_updater.get_loss_terms():
            train_metrics.add(term, iteration_metric=True)
        for term in d_crit.get_loss_terms():
            train_metrics.add(term, iteration_metric=True)
        for name in fid_metrics.keys():
            eval_metrics.add(name)
        eval_metrics.add_media_metric('samples')

        G.train(); D.train()
        iters_per_epoch = int(math.floor(len(dataloader) / n_disc_iters)) * n_disc_iters
        epochs = config['train']['step-1'].get('epochs', 11)

        for epoch in range(1, epochs):
            data_iter = iter(dataloader)
            curr_g = 0
            for i in range(1, iters_per_epoch + 1):
                data, _ = next(data_iter)
                real = data.to(device)
                d_loss, _ = train_disc(G, D, d_opt, d_crit, real, batch_size, train_metrics, device)
                if i % n_disc_iters == 0:
                    curr_g += 1
                    g_loss, _ = train_gen(g_updater, G, D, g_opt, batch_size, train_metrics, device)
                    if curr_g % config.get('log-every', 50) == 0:
                        print(f"Epoch {epoch}/{epochs-1}, G_loss={g_loss:.4f}, D_loss={d_loss:.4f}")
            # sample and eval
            G.eval()
            with torch.no_grad():
                fake = G(fixed_noise).cpu()
            G.train()
            eval_metrics.log_image('samples', group_images(fake, classifier=None, device=device))
            train_metrics.finalize_epoch()
            evaluate(G, fid_metrics, eval_metrics, batch_size, test_noise, device, None)
            eval_metrics.finalize_epoch()

        # checkpoint and prune
        run_folder = os.path.join(cp_dir, str(params.config_id))
        checkpoint_gan(G, D, g_opt, d_opt, None,
                       {'eval': eval_metrics.stats, 'train': train_metrics.stats},
                       config, output_dir=run_folder)

        final_fid = eval_metrics.stats['fid'][-1]
        if final_fid < best_so_far:
            best_so_far = final_fid
        else:
            try:
                shutil.rmtree(run_folder)
            except Exception as e:
                print(f"Failed to prune {run_folder}: {e}")
        return final_fid

    # define hyperparameters
    cs = ConfigurationSpace()
    cs.add_hyperparameters([
        Float("g_lr", (1e-4, 1e-3), default=0.0002),
        Float("d_lr", (1e-4, 1e-3), default=0.0002),
        Float("g_beta1", (0.1, 1), default=0.5),
        Float("d_beta1", (0.1, 1), default=0.5),
        Float("g_beta2", (0.1, 1), default=0.999),
        Float("d_beta2", (0.1, 1), default=0.999),
        Integer("n_blocks", (1, 5), default=3),
    ])
    scenario = Scenario(cs, deterministic=True, n_trials=1, walltime_limit=7200)
    smac = HyperparameterOptimizationFacade(scenario, train, overwrite=True)
    incumbent = smac.optimize()

    best_config = dict(incumbent)
    out_file = os.path.join(
        os.environ['FILESDIR'],
        f"step-1-best-config-{args.dataset}-bayesian-{pos_class}v{neg_class}-{run_id}.txt"
    )
    with open(out_file, 'w') as f:
        f.write(os.path.join(cp_dir, str(incumbent.config_id)))
    print(f"Saved best config to {out_file}")
    print("Best Configuration:", best_config)
    wandb.finish()


if __name__ == '__main__':
    main()
