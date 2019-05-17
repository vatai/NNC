"""TODO(vatai): time measurement
    start = time.clock()
    eval_results = evaluator(model, test_data)
    end = time.clock()

TODO(vatai): extract delta: if has '-' use partial

"""

# from functools import partial
import time
import numpy as np

import keras.datasets
from keras.layers import Dense, Conv2D
from keras.models import Model
from keras.utils import multi_gpu_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

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
    experiment_args = {
        'gpus': 1,
        'model_name': 'vgg19',
        'dataset_name': 'mnist',
        'coded_update_list': [("D", "p"), ("C2", "np")],
        'on_nth_epoch': 10,
    }
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['categorical_accuracy',
                    'top_k_categorical_accuracy'],
    }
    fit_args = {
        'epochs': 2,
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

    # Model
    model_name = experiment_args['model_name']
    model_class, preproc_dict = model_dict[model_name]
    base_model = model_class(include_top=False, pooling='avg')
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
    result = model.evaluate_generator(*test_data)
    results = dict(zip(['loss', 'top1', 'top5'], result))
    return results
