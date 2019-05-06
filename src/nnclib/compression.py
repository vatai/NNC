"""Compression related code."""

import numpy as np

from keras import optimizers
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import Callback


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


def meld(weights):
    """Return the melded weight matrix."""
    sorting = np.argsort(weights, axis=0)
    weights = np.take_along_axis(weights, sorting, axis=0)
    weights = np.mean(weights, axis=1)
    unsort = np.argsort(sorting, axis=0)
    weights = np.take_along_axis(weights[:, np.newaxis], unsort, axis=0)
    return weights


def norm(weights, melder):
    """Return weight matrix after normalised melding."""
    norms = np.linalg.norm(weights, axis=0)
    weights /= norms
    weights = melder(weights)
    weights *= norms
    return weights


def norm_meld(weights):
    """Combine normalisation and melding."""
    return norm(weights, meld)


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
    return _reshape_meld(weights, meld)


def reshape_norm_meld(weights):
    """Reshape, normalise and meld weights matrix."""
    return _reshape_meld(weights, norm_meld)


# """Keras specific."""

class BatchWeightsUpdater(Callback):
    """Batch updater callbacks."""
    def __init__(self, types, updater, on_nth_batch=0, on_nth_epoch=0):
        self.types = types
        self.updater = updater
        self._batch = on_nth_batch
        self._epoch = on_nth_epoch
        self._on_nth_batch = on_nth_batch
        self._on_nth_epoch = on_nth_epoch
        super().__init__()

    def on_batch_end(self, batch):
        """Meld weights on the end of each nth batch."""
        if batch and self._batch == batch:
            weights_updater(self.model, self.types, self.updater)
            self._batch += self._on_nth_batch

    def on_epoch_end(self, epoch):
        """Meld weights on the end of each nth epoch."""
        if epoch and self._epoch == epoch:
            weights_updater(self.model, self.types, self.updater)
            self._epoch += self._on_nth_epoch


def weights_updater(model, types, updater):
    """For every layer of the `model` which belongs to one of the `types`
    then perform `weights_updater` on it.

    """
    for layer in model.layers:
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
    from keras.utils import Sequence

    if isinstance(data, types.GeneratorType, Sequence):
        trainer_fun = model.fit_generator
    else:
        trainer_fun = model.fit

    trainer_fun(data, **fit_args)
