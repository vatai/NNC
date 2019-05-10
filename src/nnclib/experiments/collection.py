from os.path import expanduser

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Conv2D
from sacred import Experiment

from nnclib.compression import weights_updater, reshape_norm_meld
from nnclib.generators import CropGenerator

inceptionresnetv2_experiment = Experiment()


@inceptionresnetv2_experiment.config
def _inceptionresnetv2_config():
    # pylint: disable=unused-variable
    # flake8: noqa: F481
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['categorical_accuracy',
                    'top_k_categorical_accuracy'],
    }
    gen_args = {
        'img_dir': expanduser("~/tmp/ilsvrc/db"),
        'val_file': expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
        'batch_size': 32,
        'target_size': 299,
        # 'fast_mode': 1
    }
    updater_list = [(Dense, reshape_norm_meld), (Conv2D,
                                                 reshape_norm_meld)]


@inceptionresnetv2_experiment.main
def _inceptionresnetv2_main(compile_args, gen_args, updater_list):
    model = InceptionResNetV2()
    weights_updater(model, updater_list)
    model.compile(**compile_args)
    results = model.evaluate_generator(CropGenerator(**gen_args))
    return results
