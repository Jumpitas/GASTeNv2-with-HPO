import os
import yaml
from src.data_loaders import valid_dataset
from src.gan.loss import valid_loss
from schema import Schema, SchemaError, Optional, And, Or


config_schema = Schema({
    "project": str,
    "name": str,
    "out-dir": os.path.exists,
    "data-dir": os.path.exists,
    "fid-stats-path": os.path.exists,
    "fixed-noise": Or(And(str, os.path.exists), int),
    "test-noise": os.path.exists,
    Optional("compute-fid"): bool,
    Optional("device", default="cpu"): str,
    Optional("num-workers", default=0): int,
    Optional("num-runs", default=1): int,
    Optional("step-1-seeds"): [int],
    Optional("step-2-seeds"): [int],
    "dataset": {
        "name": And(str, valid_dataset),
        Optional("binary"): {"pos": int, "neg": int}
    },
    "model": {
        "z_dim": int,
        "architecture": Or(
            {
                "name": "dcgan",
                "g_filter_dim": int,
                "d_filter_dim": int,
                "g_num_blocks": int,
                "d_num_blocks": int,
            },
            {
                "name": "dcgan-v2",
                "g_filter_dim": int,
                "d_filter_dim": int,
                "g_num_blocks": int,
                "d_num_blocks": int,
            },
            {
                "name": "resnet",
                "g_filter_dim": int,
                "d_filter_dim": int,
            },
            {
                "name": "chest-xray",
                "g_filter_dim": int,
                "d_filter_dim": int,
                "g_num_blocks": int,
                "d_num_blocks": int,
            },
            {
                "name": "stl10_sagan",
                "g_filter_dim": int,
                "d_filter_dim": int,
                "g_num_blocks": int,
                "d_num_blocks": int,
            },
            {
                "name": "cifar10_sagan",
                "g_filter_dim": int,
                "d_filter_dim": int,
                "g_num_blocks": int,
                "d_num_blocks": int,
            },

            {
                "name": "imagenet",
                "g_filter_dim": int,
                "d_filter_dim": int,
                "g_num_blocks": int,
                "d_num_blocks": int,
            }
        ),
        "loss": Or(
            {
                "name": "wgan-gp",
                "args": { "lambda": float },
            },
            { "name": "ns"                        },
            {
                "name": "hinge-r1",
                "args": { "lambda": float }
            }
        )
    },
    "optimizer": {
        "lr": float,
        "beta1": Or(float, int),
        "beta2": Or(float, int),
    },
    "train": {
        "step-1": Or(And(str, os.path.exists), {
            "epochs": int,
            "checkpoint-every": int,
            "batch-size": int,
            "disc-iters": int,
            Optional("early-stop"): {
                "criteria": int,
            }
        }),
        "step-2": {
            Optional("step-1-epochs", default="best"): [Or(int, "best", "last")],
            Optional("early-stop"): {
                "criteria": int,
            },
            "epochs": int,
            "checkpoint-every": int,
            "batch-size": int,
            "disc-iters": int,
            "classifier": [And(str, os.path.exists)],
            "weight": [Or(
                {"gaussian": {"alpha": float, "var": float}},
                int,
                float,
                "mgda",
                "mgda:norm",
                {"kldiv": {"alpha": float}},
                {"gaussian-v2": {"alpha": float, "var": float}}
            )]
        }
    }
})

def read_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        # add paths
        for rel_path in ['out-dir', 'data-dir', 'fid-stats-path', 'test-noise']:
            config[rel_path] = os.environ['FILESDIR'] + '/' + config[rel_path]
        config['train']['step-2']['classifier'] = [
            os.environ['FILESDIR'] + '/' + rel_path for rel_path in config['train']['step-2']['classifier']]
        os.makedirs(config['out-dir'], exist_ok=True)
        os.makedirs(config['out-dir'], exist_ok=True)
        os.makedirs(config['data-dir'], exist_ok=True)
        # if fid-stats-path is a file, ensure its parent dir exists
        os.makedirs(os.path.dirname(config['fid-stats-path']), exist_ok=True)
        # similarly for test-noise if it’s a file
        os.makedirs(os.path.dirname(config['test-noise']), exist_ok=True)
    try:
        config_schema.validate(config)
    except SchemaError as se:
        raise se

    if "run-seeds" in config and len(config["run-seeds"]) != config["num-runs"]:
        print("Number of seeds must be equal to number of runs")
        exit(-1)

    if "run-seeds" in config["train"]["step-2"] and \
            len(config["train"]["step-2"]["run-seeds"]) != config["num-runs"]:
        print("Number of mod_gan seeds must be equal to number of runs")
        exit(-1)

    return config
