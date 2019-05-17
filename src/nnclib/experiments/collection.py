"""TODO(vatai): compression ratio

TODO(vatai): extract delta: if has '-' use partial

TODO(vatai): CHECK fix mnist https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16
"""

# from functools import partial
import numpy as np

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

def decode_updater_list(coded_update_list):
    decode_dict = {
        'D': Dense,
        'C2': Conv2D,
        'p': nnclib.compression.prune,
        'm': nnclib.compression.reshape_meldprune,
        'np': nnclib.compression.reshape_norm_prune,
        'nm': nnclib.compression.reshape_norm_meldprune
    }
    updater_list = map(lambda p: tuple(decode_dict[k] for k in p),
                       coded_update_list)
    updater_list = list(updater_list)
    return updater_list


@legion_experiment.config
def _legion_config():
    # pylint: disable=unused-variable
    # flake8: noqa: F481
    seed=42
    experiment_args = {
        'gpus': 1,
        'model_name': 'vgg19',
        'dataset_name': 'mnist',
        'coded_update_list': [("D", "p"), ("C2", "np")],
        'on_nth_epoch': 1,
    }
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['sparse_categorical_accuracy',
                    'sparse_top_k_categorical_accuracy'],
    }
    fit_args = {
        'epochs': 1,
        # 'validation_split': 0.2,
        'verbose': 2,
        'batch_size': 128,
    }


@legion_experiment.main
def _legion_main(experiment_args, compile_args, fit_args):

    # dataset
    dataset_name = experiment_args['dataset_name']
    dataset = dataset_dict[dataset_name]
    train_data, test_data = dataset.load_data()
    output_units = np.max(train_data[1]) + 1

    shape = train_data[0].shape[1:]
    if len(shape) < 3:
        shape = [-1] + list(shape) + [3]

        train_x, train_t = train_data
        test_x, test_t = test_data

        train_x = np.dstack([train_x] * 3)
        test_x = np.dstack([test_x] * 3)

        train_x = np.reshape(train_x, shape)
        test_x = np.reshape(test_x, shape)

        train_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in train_x])
        test_x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48, 48))) for im in test_x])

        train_data = train_x, train_t
        test_data = test_x, test_t

    print('>>>>', train_data[0].shape)

    # Model
    model_name = experiment_args['model_name']
    model_class, preproc_dict = model_dict[model_name]
    base_model = model_class(weights=None, include_top=False, pooling='avg')
    outputs = base_model.output
    outputs = Dense(output_units, activation='softmax')(outputs)
    model = Model(inputs=base_model.input, outputs=outputs)
    if experiment_args['gpus'] > 1:
        model = multi_gpu_model(model)

    # compile and fit
    model.compile(**compile_args)
    weights_updater_args = {
        'updater_list': decode_updater_list(experiment_args['coded_update_list']),
        'on_nth_epoch': experiment_args['on_nth_epoch']
    }
    updater = nnclib.compression.WeightsUpdater(**weights_updater_args)
    fit_args['callbacks'] = [updater]
    model.fit(*train_data, **fit_args)

    #eval
    result = model.evaluate(*test_data)
    results = dict(zip(['loss', 'top1', 'top5'], result))
    return results
