"""Model."""
import os

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
import keras.backend as K

from custom_layer_test import CompressedPrototype, get_new_weights


DEBUG = True

def insert_layer_factory(layer):
    old_weights = layer.get_weights()
    new_weights = get_new_weights(old_weights)
    print("layer.units and output_dim: {} {}"
          .format(layer.units, K.int_shape(layer.output)[1]))
    new_layer = CompressedPrototype(layer.units, weights=new_weights)
    # new_layer.set_weights(new_weights)
    return new_layer


def get_data():
    # data
    (x_train, t_train), (x_test, t_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    t_train = t_train.flatten()
    t_test = t_test.flatten()
    return (x_train, t_train), (x_test, t_test)


def get_model(train_data, hidden_units=1024, output_units=10):
    """Create a model.
    """
    x_train, t_train = train_data

    filepath = "cifar10_vgg16"
    if os.path.exists(filepath):
        model = load_model(filepath)
        return model

    # model
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       pooling='avg')
    h = base_model.output
    h = Dense(hidden_units, activation='relu')(h)
    y = Dense(output_units, activation='softmax')(h)
    model = Model(inputs=base_model.input, outputs=y)

    # trainable variables
    for layer in base_model.layers:
        layer.trainable = False

    # loss and optimizer
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    callbacks = [ModelCheckpoint(filepath, save_best_only=True)]
    # training
    model.fit(x_train, t_train, epochs=5, batch_size=64, callbacks=callbacks)
    model.save(filepath)
    return model


def modifier(model):
    # prev_out = model.input
    # for layer in model.layers:
    #     if isinstance(layer, Dense):
    #         print(layer)
    #         new_layer = CompressedPrototype(layer.units) 
    #         for x in layer.input:
    #             xA = new_layer(x)

    # return model
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if isinstance(layer, Dense):
            x = layer_input

            new_layer = insert_layer_factory(layer)
            new_layer.name = layer.name
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
        else:
            x = layer(layer_input)

        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def run(get_data, get_model, modifier):
    """Run the experiment.  This consists of getting the data, creating
    the model (including training) and evaluating the results.

    """
    train_data, test_data = get_data()
    model = get_model(train_data)
    model = modifier(model)
    print(type(test_data[0]))
    if isinstance(test_data, tuple):
        # spped up
        if DEBUG: test_data = map(lambda t: t[:10], test_data)
        result = model.evaluate(*test_data)
    else:
        msg = 'The test data is of type which can not be' + \
            ' handeled by the current implementation.'
        raise NotImplementedError(msg)
    return result


print('Results: {}'.format(run(get_data, get_model, modifier)))

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

# OOR

# train, val, test = get_generators()
# model = get_model()
# model.fit_generator(train, val)
# model.eval_generator(test)
