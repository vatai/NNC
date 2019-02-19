"""
A program to compare the acurracy of Keras models with and without
compression.

This combines the compare_with_compression.py and
get_dense_weight_size.py.  It measures accuracy and compression.
"""

import json
from os.path import expanduser, join, basename
import numpy as np

from tensorflow import set_random_seed

import keras.applications as Kapp
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D

from sacred import Experiment
# from sacred.utils import apply_backspaces_and_linefeeds # for progressbars
from sacred.observers import FileStorageObserver
from sacred.observers import TelegramObserver

from nnclib.generators import CropGenerator
from nnclib.utils import get_results_dir, model_dic


EX = Experiment()
# EX.captured_out_filter = apply_backspaces_and_linefeeds
EX.observers.append(FileStorageObserver.create(get_results_dir(__file__)))
EX.observers.append(TelegramObserver.from_config('telegram.json'))


@EX.config
def config():
    """Config function for the experiment."""
    # pylint: disable=unused-variable
    model_names = list(model_dic.keys())
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
                 'dense_smooth': 0,
                 'conv_smooth': 0,
                 'quantization': 0}
    seed = 42


def proc_weights(weights, norm, epsilon, quantization, dense_smooth, conv_smooth):
    """
    Process a single layer.

    Applies the following operations:
    """
    # II. norm
    if norm:
        norms = np.linalg.norm(weights, axis=1)
        weights /= norms[:, np.newaxis]

    # Pruning (after normalisation).
    if epsilon > 0:
        weights[np.abs(weights) < epsilon] = 0

    # III. smoothing
    if dense_smooth or conv_smooth:
        sorting = np.argsort(weights, axis=1)
        weights = np.take_along_axis(weights, sorting, axis=1)
        weights = np.mean(weights, axis=0)
        unsort = np.argsort(sorting, axis=1)
        weights = np.take_along_axis(weights[np.newaxis, :],
                                     unsort, axis=1)
    # undo: II. norm
    if norm:
        weights *= norms[:, np.newaxis]

    if quantization:
        weights = weights.astype(np.float16)

    return weights


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
    model_cls, preproc_args = model_dic[model_name]
    model = model_cls()

    nzcounts = []
    for layer in model.layers[1:]:
        if isinstance(layer, (Dense, Conv2D)):
            weights = layer.get_weights()

            # get_weights() usually returns [weights, bias] if possible we
            # don't want the bias
            # I. unpacking
            weights, rest = weights[0], weights[1:]

            shape = np.shape(weights) # old shape
            # calculate new shape and reshape weights
            height = shape[-2]
            width = shape[-1]
            for dim in shape[:-2]:
                width *= dim
            new_shape = (height, width)
            weights = np.reshape(weights, new_shape)

            weights = proc_weights(weights, **proc_args)

            # save non-zero count
            nzs = np.count_nonzero(weights, axis=1)
            nzcounts.append([nzs.shape[0], int(nzs[0])])

            # undo: reshape
            weights = np.reshape(weights, shape)
            weights = [weights] + rest

            layer.set_weights(weights)
    result = evaluate(model, preproc_args)
    return result, nzcounts


@EX.automain
def proc_all_models(_seed, model_names, proc_args):
    """
    Process all models.  Handle the results (weights = number of
    non-zero weights, accuracy) and write them in one of the results
    files.
    """

    set_random_seed(_seed)
    basedir = EX.observers[0].basedir
    print("Basedir {}\n".format(basedir))

    result_template = "_".join(
        [
            "{prefix}",
            "norm{norm}",
            "quant{quantization}",
            "dsmooth{dense_smooth}",
            "csmooth{conv_smooth}",
            "eps{epsilon}",
            "at{basedir}.json"
        ])
    accuracy_file = result_template.format(prefix="accuracy",
                                           basedir=basename(basedir),
                                           **proc_args)
    accuracy_file = join(basedir, accuracy_file)
    weights_file = result_template.format(prefix="weights",
                                          basedir=basename(basedir),
                                          **proc_args)
    weights_file = join(basedir, weights_file)

    accuracy = {}  # aggregate all results in a dictionary
    weights = {}
    for index, name in enumerate(model_names):
        result = proc_model(name)
        # If proc model returned none, then it did nothing so skip.
        if result:
            print(">>>>>> {} - {}/{} Done.".format(name, index + 1,
                                                   len(model_names)))
            print(">>>>>> {} result = {}".format(name, result[0]))
            accuracy[name] = result[0]
            weights[name] = result[1]
        json.dump(accuracy, open(accuracy_file, "w"))
        json.dump(weights, open(weights_file, "w"))
    return proc_args, accuracy, weights, basedir
