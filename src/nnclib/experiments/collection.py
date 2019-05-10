from os.path import expanduser

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sacred import Experiment

from nnclib.generators import CropGenerator

inceptionresnetv2_experiment = Experiment()


@inceptionresnetv2_experiment.config
def _inceptionresnetv2_config():
    compile_args = {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['categorical_accuracy',
                    'top_k_categorical_accuracy']
    }
    gen_args = {
        'img_dir': expanduser("~/tmp/ilsvrc/db"),
        'val_file': expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
        'batch_size': 32,
        'target_size': 299,
        'fast_mode': 1
    }


@inceptionresnetv2_experiment.main
def _inceptionresnetv2_main(compile_args, gen_args):
    model = InceptionResNetV2()
    model.compile(**compile_args)
    results = model.evaluate_generator(CropGenerator(**gen_args))
    return results
