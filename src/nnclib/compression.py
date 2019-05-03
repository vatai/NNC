"""Compression related code."""

from keras import optimizers
from keras.models import Model


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
