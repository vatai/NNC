"""
A program to compare the acurracy of Keras models with and without
compression.

This combines the compare_with_compression.py and
get_dense_weight_size.py.  It measures accuracy and compression.
"""

import json
import os.path
from os.path import expanduser
import numpy as np
import telegram

import tensorflow as tf
from tensorflow import set_random_seed

import keras.applications as Kapp
from keras.utils import multi_gpu_model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.layers.core import Dense

from sacred import Experiment
# from sacred.utils import apply_backspaces_and_linefeeds # for progressbars
from sacred.observers import FileStorageObserver
from sacred.observers import TelegramObserver

from nnclib.generators import CropGenerator
from nnclib.utils import get_results_dir


EX = Experiment()
# EX.captured_out_filter = apply_backspaces_and_linefeeds
EX.observers.append(FileStorageObserver.create(get_results_dir(__file__)))
EX.observers.append(TelegramObserver.from_config('telegram.json'))


@EX.config
def config():
    """Config function for the experiment."""
    # pylint: disable=unused-variable
    model_names = [
        "xception",  # 11
        "vgg16",     # 9
        "vgg19",     # 10
        "resnet50",  # 8
        "inceptionv3",        # 4
        "inceptionresnetv2",  # 5
        # "mobilenet",
        # "mobilenetv2",
                "densenet121",        # 1
        "densenet169",        # 2
        "densenet201",        # 3
        "nasnetmobile",       # 7
        "nasnetlarge"         # 6
    ]
    compile_args = {'optimizer': 'RMSprop',
                    'loss': 'categorical_crossentropy',
                    'metrics': [categorical_accuracy,
                                top_k_categorical_accuracy]}
    gen_args = {'img_dir': expanduser("~/tmp/ilsvrc/db"),
                'val_file': expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
                'batch_size': 32,
                'fast_mode': False}
    eval_args = {'max_queue_size': 10,
                 'workers': 4,
                 'use_multiprocessing': True,
                 'verbose': True}
    # For the no processing (original/gold results), set proc_args={}
    proc_args = {'norm': 0,
                 'epsilon': 0,
                 'smoothing': 0,
                 'quantization': 0}
    seed = 42


def get_same_type_layers(layers, ltype=Dense):
    """Return only Dense layers (or any other type)."""
    return list(filter(lambda x: isinstance(x, ltype), layers))


def proc_dense_layer(layer, norm, epsilon, quantization, smoothing):
    """Process a single layer if it is Dense (or other given type)."""
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
    nzcount = (nzs.shape[0], int(nzs[0]))
    return (dense, bias), nzcount


@EX.capture
def evaluate(model, preproc_args, compile_args, gen_args, eval_args):
    """Evaluate model."""
    model.compile(**compile_args)
    gen_args.update(preproc_args)
    generator = CropGenerator(**gen_args)
    result = model.evaluate_generator(generator, **eval_args)
    return result


@EX.capture
def proc_model(model_name, proc_args=None):
    """
    Process one model based on the model name.  If proc_args is {} or
    None then evaluate all models, as provided by keras, otherwise
    process the Dense layers using some method.
    """
    # because of sacred:
    # pylint: disable=no-value-for-parameter
    model_dic = {"xception":
                 (Kapp.xception.Xception,
                  {'preproc': Kapp.xception.preprocess_input,
                   'target_size': 299}),
                 "vgg16":
                 (Kapp.vgg16.VGG16,
                  {'preproc': Kapp.vgg16.preprocess_input,
                   'target_size': 224}),
                 "vgg19":
                 (Kapp.vgg19.VGG19,
                  {'preproc': Kapp.vgg19.preprocess_input,
                   'target_size': 224}),
                 "resnet50":
                 (Kapp.resnet50.ResNet50,
                  {'preproc': Kapp.resnet50.preprocess_input,
                   'target_size': 224}),
                 "inceptionv3":
                 (Kapp.inception_v3.InceptionV3,
                  {'preproc': Kapp.inception_v3.preprocess_input,
                   'target_size': 299}),
                 "inceptionresnetv2":
                 (Kapp.inception_resnet_v2.InceptionResNetV2,
                  {'preproc': Kapp.inception_resnet_v2.preprocess_input,
                   'target_size': 299}),
                 "mobilenet":
                 (Kapp.mobilenet.MobileNet,
                  {'preproc': Kapp.mobilenet.preprocess_input,
                   'target_size': 224}),
                 # "mobilenetv2":
                 # (Kapp.mobilenet_v2.MobileNetV2,
                 #  {'preproc': Kapp.mobilenet_v2.preprocess_input,
                 #   'target_size': 224}),
                 "densenet121":
                 (Kapp.densenet.DenseNet121,
                  {'preproc': Kapp.densenet.preprocess_input,
                   'target_size': 224}),
                 "densenet169":
                 (Kapp.densenet.DenseNet169,
                  {'preproc': Kapp.densenet.preprocess_input,
                   'target_size': 224}),
                 "densenet201":
                 (Kapp.densenet.DenseNet201,
                  {'preproc': Kapp.densenet.preprocess_input,
                   'target_size': 224}),
                 "nasnetmobile":
                 (Kapp.nasnet.NASNetMobile,
                  {'preproc': Kapp.nasnet.preprocess_input,
                   'target_size': 224}),
                 "nasnetlarge":
                 (Kapp.nasnet.NASNetLarge,
                  {'preproc': Kapp.nasnet.preprocess_input,
                   'target_size': 331})}
    model_cls, preproc_args = model_dic[model_name]
    model = model_cls()
    # try:
    #     model = multi_gpu_model(model)
    # except ValueError as exception:
    #     print(exception)

    layers = get_same_type_layers(model.layers)
    if not layers:
        # If the model has no dense layers, skip it by returning None.
        return None
    if not proc_args:
        # if proc_args == None or {} then just evaluate.
        return evaluate(model, preproc_args)
    nzcounts = []
    for layer in layers:
        new_layer, nzcount = proc_dense_layer(layer, **proc_args)
        layer.set_weights(new_layer)
        nzcounts.append(nzcount)
    result = evaluate(model, preproc_args)
    return result, nzcounts


@EX.automain
def proc_all_models(_seed, model_names, proc_args):
    """Process all models."""

    set_random_seed(_seed)
    basedir = EX.observers[0].basedir
    print("Basedir {}\n".format(basedir))
    json_name = "eval_norm{norm}_quant{quantization}_" \
        "smooth{smoothing}_eps{epsilon}.json"
    result_file = os.path.join(basedir, json_name.format(**proc_args))
    json_name = "weight_norm{norm}_quant{quantization}_" \
        "smooth{smoothing}_eps{epsilon}.json"
    weights_file = os.path.join(basedir, json_name.format(**proc_args))
    aggregation = {}  # aggregate all results in a dictionary
    weights = {}
    for index, name in enumerate(model_names):
        result = proc_model(name)
        # If proc model returned none, then it did nothing so skip.
        if result:
            print(">>>>>> {} - {}/{} Done.".format(name, index + 1,
                                                   len(model_names)))
            print(">>>>>> {} result = {}".format(name, result[0]))
            aggregation[name] = result[0]
            weights[name] = result[1]
        json.dump(aggregation, open(result_file, "w"))
        json.dump(weights, open(weights_file, "w"))
    return aggregation