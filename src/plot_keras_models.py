"""
A program to investigate pretrained models, about the distribution of
weights.
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from nnclib.utils import model_dic, reshape_weights


def get_same_type_layers(layers, ltype=(Dense, Conv2D)):
    """Return only Dense or Conv2D layers (or any other type)."""
    return list(filter(lambda x: isinstance(x, ltype), layers))


def proc_dense_layer(name, layer, idx):
    """Process a single layer if it is Dense (or other given type)."""
    weights = layer.get_weights()
    dense = weights[0]
    dense = reshape_weights(dense)
    sorted_dense = np.sort(dense, axis=1)
    norms_dense = np.linalg.norm(dense, axis=1)

    normalised = sorted_dense / norms_dense[:, np.newaxis]
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(normalised.T)
    ax[0].set_title("{}-{} - Normalised".format(name, idx))
    ax[0].set_ylim(-1, 1)
    ax[1].plot(sorted_dense.T)
    ax[1].set_title("{}-{} - Not normalised".format(name, idx))
    ax[1].set_ylim(-1, 1)
    # plt.show()
    fig.savefig("imgs/{}-{}".format(name, idx))
    plt.close(fig)


def proc_model(name="vgg16"):
    """Process one model based on the model name."""
    model = model_dic[name][0]()
    layers = get_same_type_layers(model.layers)
    for idx, layer in enumerate(layers):
        proc_dense_layer(name, layer, idx)


def proc_all_models():
    """Process all models."""
    num_models = len(model_dic.keys())
    for i, name in enumerate(model_dic.keys()):
        print("{} - {}/{}".format(name, i+1, num_models))
        proc_model(name)


proc_all_models()
print("Done")
