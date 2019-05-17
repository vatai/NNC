"""InceptionResNetV2 experiment."""

# from functools import partial
# from os.path import expanduser
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
        'd_delta': 0.005,
        'c_delta': 0.005,
    }
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['categorical_accuracy',
                    'top_k_categorical_accuracy'],
    }
    fit_args = {
        'epochs': 300,
        # 'validation_split': 0.2,
        'verbose': 2,
        'batch_size': 128,
    }
    gen_args = {
        'batch_size': 64,
        # 'fast_mode': 1,
        'target_size': 299,
        # 'preproc': preprocess_input,
    }


@legion_experiment.main
def _legion_main(experiment_args, compile_args, fit_args, gen_args):
    coded_ulist = [("D", "p"), ("C2", "np")]
    updater_list = decode_updater_list(coded_ulist)

    # dataset
    dataset_name = experiment_args['dataset_name']
    dataset = dataset_dict[dataset_name]
    train_data, test_data = dataset.load_data()
    output_units = np.max(train_data[1]) + 1

    # Model
    model_name = experiment_args['model_name']
    ModelClass, preproc_dict = model_dict[model_name]
    base_model = ModelClass(include_top=False, pooling='avg')
    outputs = base_model.output
    outputs = Dense(output_units, activation='softmax')(outputs)
    model = Model(inputs=base_model.input, outputs=outputs)
    print('>>>>>>>>>>>>', preproc_dict)
    if experiment_args['gpus'] > 1:
        model = multi_gpu_model(model)


    # weights_updater(model, updater_list)
    model.compile(**compile_args)
    # result = model.evaluate_generator(CropGenerator(**gen_args),
    #                                   **eval_args)
    # results = dict(zip(['loss', 'top1', 'top5'], result))
    results = 0
    return results
