"""
A program to compare the acurracy of Keras models with and without
compression.
"""

# TODO: move EvalGenerator to separate module
# TODO: improve evaluation, bounding boxes?
# TODO: apply for all models
# TODO: cleanup observer code

import math
import os
import keras.utils
import datetime
import numpy as np
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.layers.core import Dense
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

ex_name = os.path.basename(__file__).split('.')[0]
now = datetime.datetime.now()
ex = Experiment(ex_name)
ex.captured_out_filter = apply_backspaces_and_linefeeds
results_dir = 'results/' + ex_name + '/' + now.strftime('%Y%m%d/%H%M%S')
results_dir += '-' + str(os.getpid()) + '_' + os.uname()[1]
EX = Experiment()

EX.observers.append(FileStorageObserver.create(results_dir))


class EvalGenerator(keras.utils.Sequence):
    """
    ResNet50 evaluation Sequence, might be ok for other models as
    well.
    """
    def __init__(self, val_file, db_path, batch_size):
        self.db_path = db_path
        self.file_list = []
        self.category_list = []
        with open(val_file) as file:
            for line in file:
                img_path, cat_str = line.split(" ")
                cat = int(cat_str)
                img_path = os.path.join(db_path, img_path)
                self.file_list.append(img_path)
                self.category_list.append(cat)
        self.batch_size = batch_size
        self.indices = np.arange(len(self.file_list))

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, idx):
        # print('Evaluating idx: {}/{}'.format(idx, self.__len__()))
        inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        inputs_batch = np.zeros([self.batch_size, 224, 224, 3], np.float32)
        outputs_batch = np.zeros([self.batch_size, 1000], np.float32)
        for i, j in enumerate(inds):
            img = image.load_img(self.file_list[j], target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            inputs_batch[i] = img
            cat = self.category_list[j]
            outputs_batch[i] = keras.utils.to_categorical(cat, 1000)
        return inputs_batch, outputs_batch


@EX.config
def config():
    """Config function for the experiment."""
    # pylint: disable=unused-variable
    compile_args = {'optimizer': 'RMSprop',  # noqa: F841
                    'loss': 'categorical_crossentropy',
                    'metrics': [categorical_accuracy,
                                top_k_categorical_accuracy]}
    gen_args = {'db_path': "/home/vatai/tmp/ilsvrc/db",  # noqa: F841
                'val_file': "/home/vatai/tmp/ilsvrc/caffe_ilsvrc12/val.txt",
                'batch_size': 64}
    eval_args = {'max_queue_size': 10,  # noqa: F841
                 'workers': 1,
                 'use_multiprocessing': False}


def get_model(name: str):
    """The main procedure to be called."""
    class_dic = {"xception": keras.applications.xception.Xception,
                 "vgg16": keras.applications.vgg16.VGG16,
                 "vgg19": keras.applications.vgg19.VGG19,
                 "resnet50": keras.applications.resnet50.ResNet50,
                 "inceptionv3": keras.applications.inception_v3.InceptionV3,
                 "inceptionresnetv2":
                 keras.applications.inception_resnet_v2.InceptionResNetV2,
                 "mobilenet": keras.applications.mobilenet.MobileNet,
                 "mobilenetv2": keras.applications.mobilenet_v2.MobileNetV2,
                 "densenet121": keras.applications.densenet.DenseNet121,
                 "densenet169": keras.applications.densenet.DenseNet169,
                 "densenet201": keras.applications.densenet.DenseNet201,
                 "nasnetmobile": keras.applications.nasnet.NASNetMobile,
                 "nasnetlarge": keras.applications.nasnet.NASNetLarge}
    return class_dic[name]()


def get_same_type_layers(layers, ltype=Dense):
    """Return only Dense layers (or any other type)."""
    return list(filter(lambda x: type(x) == ltype, layers))


def proc_dense_layer(layer, norm=True):
    """Process a single layer if it is Dense (or other given type)."""
    assert type(layer) == Dense
    dense, bias = layer.get_weights()
    args = np.argsort(dense, axis=1)
    out = np.take_along_axis(dense, args, axis=1)
    norms_dense = np.linalg.norm(dense, axis=1)
    if norm:
        out /= norms_dense[:, np.newaxis]
    mean = out.mean(axis=0)
    compressed_dense = np.take_along_axis(mean[np.newaxis, :],
                                          np.argsort(args, axis=1),
                                          axis=1)
    if norm:
        compressed_dense *= norms_dense[:, np.newaxis]
    return compressed_dense, bias


@EX.capture
def evaluate(model, compile_args, gen_args, eval_args):
    """Evaluate model."""
    model.compile(**compile_args)
    generator = EvalGenerator(**gen_args)
    result = model.evaluate_generator(generator, **eval_args)
    return result


def proc_model(name):
    """Process one model based on the model name."""
    # pylint: disable=no-value-for-parameter
    model = get_model(name)
    gold = evaluate(model)
    layers = get_same_type_layers(model.layers)
    for layer in layers:
        with_norm_layer = proc_dense_layer(layer, True)
        without_norm_layer = proc_dense_layer(layer, False)
        layer.set_weights(with_norm_layer)
        with_norm = evaluate(model)
        layer.set_weights(without_norm_layer)
        without_norm = evaluate(model)
    return gold, with_norm, without_norm


@EX.automain
def proc_all_models():
    """Process all models."""
    model_names = ["xception", "vgg16", "vgg19", "resnet50", "inceptionv3",
                   "inceptionresnetv2", "mobilenet", "mobilenetv2",
                   "densenet121", "densenet169", "densenet201", "nasnetmobile",
                   "nasnetlarge"]
    for index, name in enumerate(model_names):
        print("{} - {}/{}".format(name, index, len(model_names)))
        result = proc_model(name)
        print("{} original = {}".format(name, result[0]))
        print("{} with normalisation = {}".format(name, result[1]))
        print("{} without normalisation = {}".format(name, result[2]))
