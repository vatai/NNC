"""TODO(vatai): compression ratio

TODO(vatai): CHECK extract delta: if has '-' use partial

TODO(vatai): CHECK fix mnist https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16
"""

from functools import partial
import numpy as np

from tensorflow import set_random_seed
from keras.layers import Dense, Conv2D
from keras.models import Model
from keras.preprocessing.image import img_to_array, array_to_img
from keras.utils import multi_gpu_model
from sacred import Experiment
from sacred.observers import FileStorageObserver
import keras.datasets

import nnclib.compression
from nnclib.model_dict import model_dict

inceptionresnetv2_experiment = Experiment()
observer = FileStorageObserver.create('experiment_results')
inceptionresnetv2_experiment.observers.append(observer)


legion_experiment = Experiment()
# observer = FileStorageObserver.create('experiment_results')
legion_experiment.observers.append(observer)


dataset_dict = {
    'mnist': keras.datasets.mnist,
    'fashion_mnist': keras.datasets.fashion_mnist,
    'cifar10': keras.datasets.cifar10,
    'cifar100': keras.datasets.cifar100
}

@legion_experiment.config
def _legion_config():
    # pylint: disable=unused-variable
    # flake8: noqa: F481
    seed = 42
    experiment_args = {
        'gpus': 1,
        'model_name': 'vgg19',
        'dataset_name': 'mnist',
        'coded_updater_list': [("D", "m-0.005")],
        'on_nth_epoch': 10,
    }
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['sparse_categorical_accuracy',
                    'sparse_top_k_categorical_accuracy'],
    }
    fit_args = {
        'epochs': 300,
        'shuffle': True,
        'validation_split': 0.15,
        'verbose': 1,
        'batch_size': 128,
    }


def decode_updater_list(coded_updater_list):
    decode_dict = {
        'D': Dense,
        'C2': Conv2D,
        'p': nnclib.compression.prune,
        'm': nnclib.compression.reshape_meldprune,
        'np': nnclib.compression.reshape_norm_prune,
        'nm': nnclib.compression.reshape_norm_meldprune
    }

    updater_list = []
    for typ, upd in coded_updater_list:
        layer_type = decode_dict[typ]
        if '-' in upd:
            upd = upd.split('-')
            upd[0] = decode_dict[upd[0]]
            upd[1] = float(upd[1])
            updater = partial(upd[0], delta=upd[1])
        else:
            updater = decode_dict[upd]
        updater_list.append((layer_type, updater))
    return updater_list


def _resize(im):
    im = array_to_img(im, scale=False).resize((48, 48))
    return img_to_array(im)


def fix_mnist_data(train_data, test_data):
    """Process mnist like {train,test}_data to work with
    keras.applications pretrained models.

    """
    shape = train_data[0].shape[1:]
    if len(shape) < 3:
        shape = [-1] + list(shape) + [3]

        train_x, train_t = train_data
        test_x, test_t = test_data

        train_x = np.dstack([train_x] * 3)
        test_x = np.dstack([test_x] * 3)

        train_x = np.reshape(train_x, shape)
        test_x = np.reshape(test_x, shape)

        train_x = np.asarray([_resize(im) for im in train_x])
        test_x = np.asarray([_resize(im) for im in test_x])

        train_data = train_x, train_t
        test_data = test_x, test_t

    return train_data, test_data


@legion_experiment.main
def _legion_main(_seed, experiment_args, compile_args, fit_args):
    set_random_seed(_seed)

    # dataset
    dataset_name = experiment_args['dataset_name']
    dataset = dataset_dict[dataset_name]
    train_data, test_data = dataset.load_data()
    output_units = np.max(train_data[1]) + 1
    train_data, test_data = fix_mnist_data(train_data, test_data)

    # Model
    model_name = experiment_args['model_name']
    model_class, preproc_dict = model_dict[model_name]
    print(preproc_dict)
    preprocess_input = preproc_dict['preproc']
    print(preprocess_input)
    train_data = preprocess_input(train_data[0]), train_data[1]
    test_data = preprocess_input(test_data[0]), train_data[1]
    base_model = model_class(weights='imagenet', include_top=False, pooling='avg')
    outputs = base_model.output
    outputs = Dense(output_units, activation='softmax')(outputs)
    model = Model(inputs=base_model.input, outputs=outputs)
    if experiment_args['gpus'] > 1:
        model = multi_gpu_model(model)

    # compile and fit
    model.compile(**compile_args)
    weights_updater_args = {
        'updater_list': decode_updater_list(experiment_args['coded_updater_list']),
        'on_nth_epoch': experiment_args['on_nth_epoch']
    }
    updater = nnclib.compression.WeightsUpdater(**weights_updater_args)
    fit_args['callbacks'] = [updater]
    model.fit(*train_data, **fit_args)

    #eval
    result = model.evaluate(*test_data)
    results = dict(zip(['loss', 'top1', 'top5'], result))
    return results
