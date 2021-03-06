"""
This program collects the weights of all the conv2d and dense layers
to feed it to the jupyter notebook for analysis.
"""

from os.path import join, expanduser

from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
import numpy as np

from nnclib.model_dict import model_dict
from nnclib.utils import reshape_weights


def proc_layer(layer):
    if isinstance(layer, (Dense, Conv2D)):
        # print('type', type(layer))
        if isinstance(layer, Dense):
            typ = 'dense'
        else:
            typ = 'conv2d'
        weights = layer.get_weights()[0]
        shp = np.shape(weights)
        shp = 'x'.join(map(str, shp))
        result = reshape_weights(weights)
        return [typ, shp, result]

def proc_model(model):
    results = []
    for idx, layer in enumerate(model.layers):
        result = proc_layer(layer)
        if result is not None:
            results.append([idx] + result)
    return results


def write_results(results, name, base=expanduser("~/tmp/nnc_weights")):
    for idx, typ, shp, result in results:
        file_name = "{}_{}_{}_{}".format(name, idx, typ, shp)
        file_name = join(base, file_name)
        np.save(file_name, result)


def proc_all_models():
    for name, value in model_dict.items():
        print(name)
        model = value[0]()
        results = proc_model(model)
        write_results(results, name)


if __name__ == '__main__':
    proc_all_models()
