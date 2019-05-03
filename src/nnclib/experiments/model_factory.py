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


def vgg16_mod(train_data, hidden_units=1024, output_units=10):
    """Modified vgg16

    """
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

    # loss and optimiser
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    # training
    if train_data:
        x_train, t_train = train_data
        kwargs = {
            'epochs': 5,
            'batch_size': 64,
            'validation_split': 0.1,
            # 'val_loss': categorical_crossentropy,
            'callbacks': [ModelCheckpoint(filepath, save_best_only=True)],
            
        }
        model.fit(x_train, t_train, **kwargs)
        model.save(filepath)
    return model
