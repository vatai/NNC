"""Model factories: functions which return a model to be used by
`run_experiment`.

"""

import os

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.models import Model, load_model
from keras.utils import multi_gpu_model


def vgg16_mod(compile_args, num_gpus=1, hidden_units=4046,
              output_units=10):
    """Modified vgg16. """

    # model
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       pooling='avg')
    h = base_model.output
    h = Dense(hidden_units, activation='relu')(h)
    y = Dense(output_units, activation='softmax')(h)
    model = Model(inputs=base_model.input, outputs=y)

    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus)

    # trainable variables
    for layer in base_model.layers:
        layer.trainable = False

    # some defaults
    if 'loss' not in compile_args:
        compile_args['loss'] = 'sparse_categorical_crossentropy'
    if 'optimizer' not in compile_args:
        compile_args['optimizer'] = optimizers.SGD(lr=0.01)
    if 'metrics' not in compile_args:
        compile_args['metrics'] = ['accuracy']

    model.compile(**compile_args)

    return model
