"""Model factories: functions which return a model to be used by
`run_experiment`.

"""

import os

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras.losses import categorical_crossentropy


def vgg16_mod(train_data, hidden_units=4046, output_units=10, compile_args=None):
    """Modified vgg16. """

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

    # some defaults
    if compile_args=None:
        compile_args = {}
    if 'loss' not in compile_args:
        compile_args['loss'] = 'sparse_categorical_crossentropy'
    if 'optimizer' not in compile_args:
        compile_args['optimizer'] = optimizers.SGD(lr=0.01)
    if 'metrics' not in compile_args:
        compile_args['metrics'] = ['accuracy']

    model.compile(**compile_args)

    # training
    if train_data:
        x_train, t_train = train_data
        fit_args = {
            'epochs': 256,
            'batch_size': 128,
            'validation_split': 0.2,
            'callbacks': [ModelCheckpoint(filepath, save_best_only=True)],
        }
        model.fit(x_train, t_train, **fit_args)
        model.save(filepath)
    return model
