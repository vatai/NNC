"""Experiment prototype, to try out the library structure.

TODO(vatai): modifier compile_args.

TODO(vatai): data generation: design matrix vs generator

TODO(vatai): instead of condition, new_layer_factory make just a
conditional_layer_factory.

"""
from functools import partial
from pprint import pprint
import os
import sys

from keras.layers import Dense

from custom_layer_test import CompressedPrototype, get_new_weights
from nnclib.experiments import run_experiment, model_factory, data_factory
from nnclib.compression import modifier

if os.path.exists('src'):
    sys.path.append('src')


def create_meld_dense(layer):
    """Called for replacing a dense layer with a 'meld dense' layer."""
    old_weights = layer.get_weights()
    new_weights = get_new_weights(old_weights)
    # print("layer.units and output_dim: {} {}"
    #       .format(layer.units, K.int_shape(layer.output)[1]))
    new_layer = CompressedPrototype(layer.units, weights=new_weights)
    # new_layer.set_weights(new_weights)
    return new_layer


def partial_isinstance(typ):
    """Partially applies isinstance(., typ)."""
    return lambda x: isinstance(x, typ)


RESULT = run_experiment(data_factory.cifar10_float32,
                        model_factory.vgg16_mod,
                        partial(modifier,
                                condition=partial_isinstance(Dense),
                                new_layer_factory=create_meld_dense))

pprint(RESULT)
