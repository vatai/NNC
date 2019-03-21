"""
Make table of epsilons from weights.  Besically, for each model
calculate the max and the average epsilon.

The max and avg epsilon is the max and avg of the layer epsilon.

The layer epsilon is the max of the column epsilon.

The column epsilon is the max - min in the column.

NOTE: These are the actual epsilons from the paper, i.e. "epsilon close" epsilons, not the pruning deltas.
"""

import glob
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt


def make_table(d):
    for model_name, data in d.items():
        total = 0
        esum, nsum, emax, nmax = 0, 0, 0, 0
        for eps, neps in data:
            emax = max(emax, eps.max())
            nmax = max(nmax, neps.max())
            esum += np.sum(eps)
            nsum += np.sum(neps)
            total += len(eps)
        line = [model_name, emax, esum / total, nmax, nsum / total]
        line = "{:17} & {:7.5} & {:9.5} & {:7.5} & {:9.5} \\\\".format(*line)
        print(line)


def layer_epsilon(layer):
    norm = np.linalg.norm(layer, axis=0)
    nlayer = layer / norm
    nlayer.sort(axis=0)
    layer.sort(axis=0)
    # eps = layer.max(axis=1) - layer.min(axis=1)
    # neps = nlayer.max(axis=1) - nlayer.min(axis=1)
    eps = np.max(np.abs(layer.T - layer.mean(axis=1)), axis=0)
    neps = np.max(np.abs(nlayer.T - nlayer.mean(axis=1)), axis=0)
    # print(layer.shape)
    # print(norm.shape)
    # print(eps.shape)
    # plt.plot(eps)
    # plt.plot(neps)
    # plt.plot(layer)
    # plt.plot(nlayer+0.5)
    # plt.legend(['e', 'n'])
    # plt.show()
    # exit()
    return eps, neps


def proc_all(base="*"):
    files = sorted(glob.glob(base))
    d = {}
    n = len(files)
    for idx, file in enumerate(files):
        layer = np.load(file)
        layer_eps = layer_epsilon(layer)
        if idx % 100 == 99:
            print("{}/{} done".format(idx+1, n))
        model = file.split("_")[0]
        if model in d:
            d[model].append(layer_eps)
        else:
            d[model] = [layer_eps]
    return d


if __name__ == '__main__':
    layer_epsilons = proc_all()
    make_table(layer_epsilons)
