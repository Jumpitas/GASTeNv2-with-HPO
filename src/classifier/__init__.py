from .simple_cnn import Classifier as SimpleCNN
from .my_mlp import Classifier as MyMLP


def construct_classifier(params, device=None):
    if params['type'] == 'cnn':
        C = SimpleCNN(params['img_size'], [params['nf'], params['nf'] * 2],
                      params['n_classes'])
    elif params['type'] == 'mlp':
        C = MyMLP(params['img_size'], params['n_classes'], [params['nf'], params['nf'] * 2])
    else:
        exit(-1)

    return C.to(device)
