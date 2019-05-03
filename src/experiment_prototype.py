"""Model."""
import os
import sys
if os.path.exists('src'):
    sys.path.append('src')

from keras import optimizers
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.datasets import cifar10
from keras.layers import Dense
import keras.backend as K


DEBUG = True


def create_new_layer(layer):
    """TODO(vatai): rename to compressed_dense()"""
    old_weights = layer.get_weights()
    new_weights = get_new_weights(old_weights)
    # print("layer.units and output_dim: {} {}"
    #       .format(layer.units, K.int_shape(layer.output)[1]))
    new_layer = CompressedPrototype(layer.units, weights=new_weights)
    # new_layer.set_weights(new_weights)
    return new_layer


def get_cifar10_float32():
    # data
    """get_cifar10_float32"""
    (x_train, t_train), (x_test, t_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    t_train = t_train.flatten()
    t_test = t_test.flatten()
    return (x_train, t_train), (x_test, t_test)


def run_experiment(get_data, get_model, modifier):
    """Run the experiment.  This consists of getting the data, creating
    the model (including training) and evaluating the results.

    """
    train_data, test_data = get_data()
    model = get_model(train_data)
    model = modifier(model)
    if isinstance(test_data, tuple):
        # sped up
        if DEBUG: test_data = map(lambda t: t[:10], test_data)
        result = model.evaluate(*test_data)
    else:
        msg = 'The test data is of type which can not be' + \
            ' handeled by the current implementation.'
        raise NotImplementedError(msg)
    return result


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


# train, val, test = get_data()
# model = get_model()
# model.fit(train, val)
# if modifier:
#     model = modifier(model)
# results = model.evaluate(data)
# save(results, model)
# # later
# data = analyse(results, model)
# write_latex_table(data)

# OR

# train, val, test = get_generators()
# model = get_model()
# model.fit_generator(train, val)
# model.eval_generator(test)

"""TODO(vatai): modifier compile_args.

TODO(vatai): naming convention.

"""
# from nnclib.experiments import run_experiment, data_factory, model_factory

# run_esperiment(data_factory.cifar10_float32, model_factory.vgg16, mod)

from functools import partial

from nnclib.generators import CropGenerator
from nnclib.experiments.model_factory import vgg16_mod
from custom_layer_test import CompressedPrototype, get_new_weights


MOD = partial(modifier,
                 condition=partial_isinstance(Dense),
                 new_layer_factory=create_new_layer)

RESULT = run_experiment(get_cifar10_float32, vgg16_mod, MOD)

print('Results: {}'.format(RESULT))
