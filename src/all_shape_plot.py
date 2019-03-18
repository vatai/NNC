"""
Make (normalised) plots for all layers for a given network
demonstrating the possibility of compression.

The plots are saved in the working directory.
"""
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from nnclib.utils import model_dict, reshape_weights


def proc_weights(p):
    a = np.argsort(p, axis=0)
    p = np.take_along_axis(p, a, axis=0)
    # uncomment the following line to skip normalisation
    p /= np.linalg.norm(p, axis=0)
    fig, ax = plt.subplots()
    ax.plot(p)
    return fig


def process(model_name):
    print(model_name)
    model = model_dict[model_name][0]()

    # skip input layer
    for idx, layer in enumerate(model.layers[1:]):
        # print(type(layer))
        weights = layer.get_weights()
        if isinstance(weights, list) and weights:
            weights = weights[0]
        shape = np.shape(weights)
        # print(shape)
        if len(shape) > 1:
            weights = reshape_weights(weights)
            fig = proc_weights(weights)
            shape_str = "x".join(map(str, shape))
            typ = str(type(layer))
            typ = typ.split('.')[-1][:-2]
            fig_name = "_".join([model_name, typ, shape_str, str(idx)])
            fig.savefig(fig_name + ".pdf")
            fig.savefig(fig_name + ".png")
            plt.close(fig)


pool = multiprocessing.Pool()
pool.map(process, model_dict.keys())
