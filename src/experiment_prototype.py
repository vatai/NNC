"""Experiment prototype, to try out the library structure.

TODO(vatai): modifier compile_args.

TODO(vatai): naming convention.

TODO(vatai): data generation: design matrix vs generator

TODO(vatai): instead of condition, new_layer_factory make just a
conditional_layer_factory.

"""
from functools import partial
import os
import sys

from keras import optimizers
from keras.layers import Dense
from keras.models import Model

from custom_layer_test import CompressedPrototype, get_new_weights
from nnclib.experiments import run_experiment, model_factory, data_factory

if os.path.exists('src'):
    sys.path.append('src')


def create_new_layer(layer):
    """TODO(vatai): rename to compressed_dense()"""
    old_weights = layer.get_weights()
    new_weights = get_new_weights(old_weights)
    # print("layer.units and output_dim: {} {}"
    #       .format(layer.units, K.int_shape(layer.output)[1]))
    new_layer = CompressedPrototype(layer.units, weights=new_weights)
    # new_layer.set_weights(new_weights)
    return new_layer


def modifier(model, condition, new_layer_factory):
    """Modifies a model: applies `condition()` to each layer in the model
    and substitutes it with a new layer obtained from
    `new_layer_factory`

    Args:

        model: the original model.

        condition: a function/functor, which if evaluates to True on a
            layer triggers the given layer to be substituted with a
            new one.

        new_layer_factory: the function which creates a new layer from
            the old one.

    Returns: model: the modified model.

    """
    # From: https://stackoverflow.com/a/54517478/568735

    # Auxiliary dictionary to describe the network graph
    input_layers_of = {}
    new_output_tensor_of = {}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in input_layers_of:
                input_layers_of.update({layer_name: [layer.name]})
            else:
                input_layers_of[layer_name].append(layer.name)

    # Set the output tensor of the input layer
    new_output_tensor_of.update({model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [new_output_tensor_of[layer_aux]
                       for layer_aux in input_layers_of[layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if condition(layer):
            x = layer_input

            new_layer = new_layer_factory(layer)
            new_layer.name = layer.name
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
        else:
            x = layer(layer_input)

        new_output_tensor_of.update({layer.name: x})

    model = Model(inputs=model.inputs, outputs=x)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizers.SGD(lr=0.01),
                metrics=['accuracy'])
    return model


def partial_isinstance(typ):
    """Partially applies isinstance(., typ).

    Args:

        `typ` (type, list): a typ or list of types passed as an the
        second arg of isinstance().

    Returns:
        function: See above.

    """
    return lambda x: isinstance(x, typ)


RESULT = run_experiment(data_factory.cifar10_float32,
                        model_factory.vgg16_mod,
                        partial(modifier,
                                condition=partial_isinstance(Dense),
                                new_layer_factory=create_new_layer))

print('Results: {}'.format(RESULT))
