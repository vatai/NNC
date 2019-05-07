"""Compression related code."""

import numpy as np

from keras import optimizers
from keras.callbacks import Callback
from keras.models import Model
from keras.utils import Sequence
# from keras.layers import Dense, Conv2D


def reshape_weights(weights):
    """
    Takes a :math:`d_1 \\times d_2 \\times \\ldots \\times d_{n-1}
    \\times d_n` dimensional tensor, and reshapes it to a :math:`d_1
    \\cdots d_{n-2} \\cdot d_n \\times d_{n-1}` dimensional matrix.
    """ 
    import numpy as np
    shape = np.shape(weights)  # old shape
    # calculate new shape and reshape weights
    height = shape[-2]
    width = shape[-1]
    for dim in shape[:-2]:
        width *= dim
    new_shape = (height, width)
    weights = np.reshape(weights, new_shape)
    return weights


def _meld(weights):
    """Return the melded weight matrix."""
    sorting = np.argsort(weights, axis=0)
    weights = np.take_along_axis(weights, sorting, axis=0)
    weights = np.mean(weights, axis=1)
    unsort = np.argsort(sorting, axis=0)
    weights = np.take_along_axis(weights[:, np.newaxis], unsort, axis=0)
    return weights


def _norm(weights, melder):
    """Return weight matrix after normalised melding."""
    norms = np.linalg.norm(weights, axis=0)
    weights /= norms
    weights = melder(weights)
    weights *= norms
    return weights


def _norm_meld(weights):
    """Combine normalisation and melding."""
    return _norm(weights, _meld)


def _reshape_check(weights, melder):
    """Reshape and apply `melder` to the weights matrix."""
    shape = np.shape(weights)
    if len(shape) > 1:
        weights = reshape_weights(weights)
        weights = melder(weights)
        weights = np.reshape(weights, shape)
    return weights


def reshape_meld(weights):
    """Reshape and meld weights matrix."""
    return _reshape_check(weights, _meld)


def reshape_norm_meld(weights):
    """Reshape, normalise and meld weights matrix."""
    return _reshape_check(weights, _norm_meld)


# """Keras specific."""

class WeightsUpdater(Callback):
    """Batch updater callbacks."""
    def __init__(self, updater_list, on_nth_batch=0, on_nth_epoch=0):
        self.updater_list = updater_list
        self._on_nth_batch = on_nth_batch
        self._on_nth_epoch = on_nth_epoch
        # `self._batch_counter` and `self._epoch_counter` are the
        # counters to keep track of when to apply the `on_batch_end()`
        # and `on_epoch_end()`. The `-1` is to ensure correct
        # behaviour (since batches and epochs are enumerated from 0),
        # and also a nifty trick to avoid calling the callback if
        # `on_nth_{batch,epoch}` is 0.
        self._batch_counter = on_nth_batch - 1
        self._epoch_counter = on_nth_epoch - 1
        super().__init__()

    def on_batch_end(self, batch, log={}):
        """Meld weights on the end of each nth batch."""
        if self._batch_counter == batch:
            weights_updater(self.model, self.updater_list)
            self._batch_counter += self._on_nth_batch

    def on_epoch_end(self, epoch, log={}):
        """Meld weights on the end of each nth epoch."""
        print("on_epoch_end({})".format(epoch))
        if self._epoch_counter == epoch:
            weights_updater(self.model, self.updater_list)
            self._epoch_counter += self._on_nth_epoch


def weights_updater(model, updater_list):
    """For every layer of the `model` which belongs to one of the `types`
    then perform `weights_updater` on it.

    """
    print("Updating model: {}".format(updater_list))

    for layer in model.layers:
        for types, updater in updater_list:
            if isinstance(layer, types):
                weights_list = layer.get_weights()
                for i, weights in enumerate(weights_list):
                    new_weights = updater(weights)
                    weights_list[i] = new_weights
                layer.set_weights(weights_list)


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


def trainer(model, data, **fit_args):
    """Keras trainer with melding."""
    import types

    if isinstance(data, (types.GeneratorType, Sequence)):
        trainer_fun = model.fit_generator
    else:
        trainer_fun = model.fit

    trainer_fun(*data, **fit_args)


def evaluator(model, data, **eval_args):
    """Keras evaluation."""
    import types

    if isinstance(data, (types.GeneratorType, Sequence)):
        eval_fun = model.eval_generator
    elif isinstance(data, tuple):
        eval_fun = model.evaluate
    else:
        msg = 'The test data is of type which can not be' + \
            ' handeled by the current implementation.'
        raise NotImplementedError(msg)

    return eval_fun(*data, **eval_args)


# def norm_meld_trainer(model, data):
#     epoch_melder = WeightsUpdater(types=(Dense, Conv2D),
#                                   updater=reshape_norm_meld,
#                                   on_nth_epoch=2)
#     trainer(model, data, callbacks=[epoch_melder])
