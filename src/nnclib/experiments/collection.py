"""InceptionResNetV2 experiment."""

from functools import partial
from os.path import expanduser

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Dense, Conv2D
from keras.utils import multi_gpu_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

from nnclib.compression import weights_updater, reshape_norm_meld
from nnclib.generators import CropGenerator

inceptionresnetv2_experiment = Experiment()
observer = FileStorageObserver.create('experiment_results')
inceptionresnetv2_experiment.observers.append(observer)


@inceptionresnetv2_experiment.config
def _inceptionresnetv2_config():
    # pylint: disable=unused-variable
    # flake8: noqa: F481
    d_delta = 0.005
    c_delta = 0.005
    updater_list = [(Dense, partial(reshape_norm_meld, delta=d_delta)),
                    (Conv2D, partial(reshape_norm_meld, delta=c_delta))]
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['categorical_accuracy',
                    'top_k_categorical_accuracy'],
    }
    gen_args = {
        'img_dir': expanduser("~/tmp/ilsvrc/db"),
        'val_file': expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
        'batch_size': 64,
        # 'fast_mode': 1,
        'target_size': 299,
        'preproc': preprocess_input,
    }
    eval_args = {
        'use_multiprocessing': False,
        'workers': 14,
        'max_queue_size': 14,
        'verbose': True,
    }
    gpus = 1


@inceptionresnetv2_experiment.main
def _inceptionresnetv2_main(gpus, compile_args, gen_args, eval_args,
                            updater_list):
    model = InceptionResNetV2()
    if gpus > 1:
        model = multi_gpu_model(model)
    weights_updater(model, updater_list)
    model.compile(**compile_args)
    result = model.evaluate_generator(CropGenerator(**gen_args),
                                      **eval_args)
    results = dict(zip(['loss', 'top1', 'top5'], result))
    return results
