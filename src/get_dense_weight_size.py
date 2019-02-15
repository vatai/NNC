"""
This program outputs the size of the dense layers in each keras
pretrained model.
"""

import os
import json
import numpy as np
from tensorflow import set_random_seed
import keras.applications as Kapp
from keras.layers.core import Dense
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import TelegramObserver

from nnclib.utils import get_results_dir

EX = Experiment()
EX.observers.append(FileStorageObserver.create(get_results_dir(__file__)))
EX.observers.append(TelegramObserver.from_config('telegram.json'))


@EX.config
def config():
    model_names = ["xception", "vgg16", "vgg19", "resnet50",
                   "inceptionv3", "inceptionresnetv2", "mobilenet",
                   # "mobilenetv2", 
                   "densenet121", "densenet169",
                   "densenet201", "nasnetmobile", "nasnetlarge"]
    proc_args = {'norm': False,
                 'epsilon': 0,
                 'smoothing': 0,
                 'quantization': 0}
    seed = 42


def get_same_type_layers(layers, ltype=Dense):
    """Return only Dense layers (or any other type)."""
    return list(filter(lambda x: isinstance(x, ltype), layers))


def proc_dense_layer(layer, norm, epsilon, quantization, smoothing):
    """
    Return the number of non-zero elements in a layer after
    processing.
    TODO finish.
    """
    assert isinstance(layer, Dense)
    dense, bias = layer.get_weights()
    if norm:
        norms = np.linalg.norm(dense, axis=1)
        dense /= norms[:, np.newaxis]
    if smoothing:
        indices = np.argsort(dense, axis=1)
        dense = np.take_along_axis(dense, indices, axis=1)
        mean = dense.mean(axis=0)
        unsort_indices = np.argsort(indices, axis=1)
        dense = np.take_along_axis(mean[np.newaxis, :],
                                   unsort_indices,
                                   axis=1)
    if epsilon != 0:
        cond = np.abs(dense) < epsilon
        dense[cond] = 0
    if norm:
        dense *= norms[:, np.newaxis]
    if quantization:
        dense = dense.astype(np.float16)
    
    nzs = np.count_nonzero(dense, axis=1)
    return (nzs.shape[0], int(nzs[0]))


@EX.capture
def proc_model(name, proc_args=None):
    """Process one model based on the model name."""
    # because of sacred:
    # pylint: disable=no-value-for-parameter
    model_dic = {"xception":
                 Kapp.xception.Xception,
                 "vgg16":
                 Kapp.vgg16.VGG16,
                 "vgg19":
                 Kapp.vgg19.VGG19,
                 "resnet50":
                 Kapp.resnet50.ResNet50,
                 "inceptionv3":
                 Kapp.inception_v3.InceptionV3,
                 "inceptionresnetv2":
                 Kapp.inception_resnet_v2.InceptionResNetV2,
                 "mobilenet":
                 Kapp.mobilenet.MobileNet,
                 # "mobilenetv2":
                 # Kapp.mobilenet_v2.MobileNetV2,
                 "densenet121":
                 Kapp.densenet.DenseNet121,
                 "densenet169":
                 Kapp.densenet.DenseNet169,
                 "densenet201":
                 Kapp.densenet.DenseNet201,
                 "nasnetmobile":
                 Kapp.nasnet.NASNetMobile,
                 "nasnetlarge":
                 Kapp.nasnet.NASNetLarge}
    model_cls = model_dic[name]
    model = model_cls()
    layers = get_same_type_layers(model.layers)
    if not layers:
        # If the model has no dense layers, skip it by returning None.
        return None
    result = []
    for layer in layers:
        weights = layer.get_weights()
        shape = weights[0].shape
        after_compression = proc_dense_layer(layer, **proc_args)
        layer_result = (shape, after_compression)
        result.append(layer_result)
    return result


@EX.automain
def proc_all_models(_seed, model_names, proc_args):
    """Process all models: print and store results."""
    set_random_seed(_seed)
    basedir = EX.observers[0].basedir
    json_name = "weight_norm{norm}_quant{quantization}_" \
        "smooth{smoothing}_eps{epsilon}.json"
    result_file = os.path.join(basedir, json_name.format(**proc_args))
    result = {}
    for index, name in enumerate(model_names):
        print(">>>>>> {} - {}/{}".format(name, index + 1, len(model_names)))
        model_result = proc_model(name)
        if model_result:
            result[name] = model_result
        json.dump(result, open(result_file, 'w'))
    return result
