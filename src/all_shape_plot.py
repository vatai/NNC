"""Make plots for all layers for a given network."""
import sys
import multiprocessing
# sys.path.append("../src")

import numpy as np
import matplotlib.pyplot as plt
from nnclib.utils import model_dict, reshape_weights

def proc_weights(p):
    a = np.argsort(p, axis=0)
    p = np.take_along_axis(p, a, axis=0)
    norm = np.linalg.norm(p, axis=0)
    p /= norm
    fig, ax = plt.subplots()
    ax.plot(p)
    return fig


def process(model_name):
    # model_name = model_info[0]
    print(model_name)
    # model = model_info[1][0]()
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
            sh1 = shape[-2]
            sh0 = shape[-1]
            for d in shape[:-2]:
                sh0 *= d
            weights = np.reshape(weights, [sh0, sh1])
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
