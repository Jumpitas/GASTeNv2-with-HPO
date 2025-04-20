import itertools
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from dotenv import load_dotenv
from utils import begin_classifier

load_dotenv()
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-dir', dest='dataroot',
                    default=f"{os.environ['FILESDIR']}/data", help='Dir with dataset')
parser.add_argument('--out-dir', dest='out_dir',
                    default=f"{os.environ['FILESDIR']}/models", help='Path to generated files')
parser.add_argument('--dataset', dest='dataset',
                    default='mnist', help='Dataset (mnist or fashion-mnist or cifar10 or stl10 or chest-xray)')
parser.add_argument('--n-classes', dest='n_classes',
                    type=int, default=10, help='Number of classes in dataset')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--batch-size', dest='batch_size',
                    type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='ADAM opt learning rate')

parser.add_argument('--pos', dest='pos_class', default=9,
                    type=int, help='Positive class for binary classification')
parser.add_argument('--neg', dest='neg_class', default=4,
                    type=int, help='Negative class for binary classification')

parser.add_argument('--epochs', type=str, default="3",
                    help='Comma-separated list of number of epochs to train for')
parser.add_argument('--classifier-type', dest='clf_type',
                    type=str, help='List with elements "cnn" or "mlp"', default='cnn')
parser.add_argument('--nf', type=str, default="2,4,8,16",
                    help='Comma-separated list of possible num features')
parser.add_argument("--seed", type=int, help='Random seed to use in generation', default=None)
parser.add_argument('--early-acc', dest='early_acc',
                    type=float, default=1.0, help='Early accuracy criteria')

def main():
    args = parser.parse_args()
    print("Arguments:", args)

    # Ensure n_classes is an integer
    try:
        n_classes = int(args.n_classes)
    except ValueError:
        n_classes = 10

    # Process epochs and classifier types into sorted unique lists
    l_epochs = list(set([e for e in args.epochs.split(",") if e.isdigit()]))
    l_clf_type = list(set([ct for ct in args.clf_type.split(",")]))
    l_epochs.sort(key=int)
    l_clf_type.sort()

    print("Epochs list:", l_epochs)
    print("Classifier types list:", l_clf_type)

    # Create an iterator over class pairs
    if args.pos_class is not None and args.neg_class is not None:
        iterator = iter([(str(args.neg_class), str(args.pos_class))])
        print("Using fixed binary pair: ({} vs {})".format(args.neg_class, args.pos_class))
    else:
        iterator = itertools.combinations(range(n_classes), 2)
        print("Using all combinations for {} classes".format(n_classes))

    # Always use begin_classifier regardless of input type
    for clf_type in l_clf_type:
        print("Processing classifier type:", clf_type)
        begin_classifier(iterator, clf_type, l_epochs, args)

if __name__ == '__main__':
    main()
