"""Collect the shapes of the Dense layer kernels, and store them in
json files.

"""

from json import dump
from multiprocessing import Pool

from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense

from nnclib.model_dict import model_dict
from nnclib.utils import reshape_weights

JSON = "dense_weights.json"


def proc(name):
    model = model_dict[name][0]()
    layers = filter(lambda x: isinstance(x, (Dense, Conv2D)), model.layers)
    if layers:
        for layer in layers:
            weights = layer.get_weights()[0]
            if isinstance(layer, Conv2D):
                weights = reshape_weights(weights)
            result.append(weights.shape)
        return (name, result)


if __name__ == '__main__':
    pool = Pool()
    result = pool.map(proc, model_dict.keys())
    d = {}
    d.update(result)
    dump(d, open(JSON, 'w'))
