#!/usr/bin/env python
import os
import sys
import itertools
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dotenv import load_dotenv
from utils import begin_classifier

def main():
    load_dotenv()

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-dir', dest='dataroot',
        default=f"{os.environ.get('FILESDIR','.')}/data",
        help='Root directory containing the dataset'
    )
    parser.add_argument(
        '--out-dir', dest='out_dir',
        default=f"{os.environ.get('FILESDIR','.')}/models",
        help='Directory where trained classifiers will be saved'
    )
    parser.add_argument(
        '--dataset', dest='dataset',
        choices=['mnist','fashion-mnist','cifar10','stl10','chest-xray','imagenet'],
        required=True,
        help='Name of the dataset'
    )
    parser.add_argument(
        '--n-classes', dest='n_classes',
        type=int, default=10,
        help='Total number of classes in the dataset'
    )
    parser.add_argument(
        '--device', dest='device',
        default='cpu',
        help='Torch device (e.g. "cuda:0" or "cpu")'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size',
        type=int, default=64,
        help='Mini-batch size for classifier training'
    )
    parser.add_argument(
        '--lr', dest='lr',
        type=float, default=1e-3,
        help='Learning rate for Adam optimizer'
    )
    parser.add_argument(
        '--pos', dest='pos_class',
        type=int, default=None,
        help='Positive class label (for binary)'
    )
    parser.add_argument(
        '--neg', dest='neg_class',
        type=int, default=None,
        help='Negative class label (for binary)'
    )
    parser.add_argument(
        '--epochs', dest='epochs',
        type=str, default="10",
        help='Comma-separated list of epoch counts to train for'
    )
    parser.add_argument(
        '--classifier-type', dest='clf_type',
        type=str, default='cnn',
        help='Comma-separated list of classifier types, e.g. "cnn,mlp,resnet18_finetune"'
    )
    parser.add_argument(
        '--nf', dest='nf',
        type=str, default="32,64,128",
        help='Comma-separated filter sizes (for cnn variants)'
    )
    parser.add_argument(
        '--seed', dest='seed',
        type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--early-acc', dest='early_acc',
        type=float, default=1.0,
        help='Early stopping accuracy threshold'
    )

    args = parser.parse_args()
    print("Arguments:", args)

    # Determine output subdirectory based on dataset and class-pair
    if args.pos_class is not None and args.neg_class is not None:
        pair = f"{args.pos_class}v{args.neg_class}"
        class_pairs = [(args.neg_class, args.pos_class)]
        print(f"Using fixed binary pair: {class_pairs[0]}")
    else:
        pair = "all_pairs"
        class_pairs = list(itertools.combinations(range(args.n_classes), 2))
        print(f"Using all {len(class_pairs)} class-pairs for {args.n_classes} classes")

    out_subdir = os.path.join(args.out_dir, args.dataset, pair)
    os.makedirs(out_subdir, exist_ok=True)
    print(f"→ Saving classifiers under: {out_subdir}")
    args.out_dir = out_subdir

    # Parse epochs and classifier types
    epochs_list = sorted({int(e) for e in args.epochs.split(',') if e.isdigit()})
    print("Epochs to run:", epochs_list)
    clf_types = sorted({ct.strip() for ct in args.clf_type.split(',') if ct.strip()})
    print("Classifier types:", clf_types)
    nf_list = sorted({nf for nf in args.nf.split(',') if nf.isdigit()}, key=int)
    print("Feature-dims (nf) to try:", nf_list)

    # Launch training runs
    for clf_type in clf_types:
        for neg, pos in class_pairs:
            print(f"\n=== Training {clf_type} on {args.dataset} [{pos} vs {neg}] ===")
            begin_classifier(
                iterator=iter([(neg, pos)]),
                clf_type=clf_type,
                l_epochs=epochs_list,
                args=args
            )

if __name__ == '__main__':
    main()
