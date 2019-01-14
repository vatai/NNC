"""
A program to compare the acurracy of Keras models with and without
compression.
"""

# TODO: TEST Compress by sparsification (implemented in proc_dense_layer)
# TODO: Closses difference

import json
import os.path
import numpy as np
import telegram
import keras.applications as Kapp
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.layers.core import Dense
from sacred import Experiment
# from sacred.utils import apply_backspaces_and_linefeeds # for progressbars
from sacred.observers import FileStorageObserver
from sacred.observers import TelegramObserver
from generators import CropGenerator
from utils import get_results_dir


EX = Experiment()
# EX.captured_out_filter = apply_backspaces_and_linefeeds
EX.observers.append(FileStorageObserver.create(get_results_dir(__file__)))
EX.observers.append(TelegramObserver.from_config('telegram.json'))


@EX.config
def config():
    """Config function for the experiment."""
    # pylint: disable=unused-variable
    json_name = 'results'
    compile_args = {'optimizer': 'RMSprop',
                    'loss': 'categorical_crossentropy',
                    'metrics': [categorical_accuracy,
                                top_k_categorical_accuracy]}
    gen_args = {'img_dir': "/home/vatai/tmp/ilsvrc/db",
                'val_file': "/home/vatai/tmp/ilsvrc/caffe_ilsvrc12/val.txt",
                'batch_size': 64,
                'fast_mode': False}
    eval_args = {'max_queue_size': 10,
                 'workers': 1,
                 'use_multiprocessing': False}
    # For the no processing (original/gold results), set proc_args={}
    proc_args = {'norm': False,
                 'epsilon': 0}


def get_same_type_layers(layers, ltype=Dense):
    """Return only Dense layers (or any other type)."""
    return list(filter(lambda x: isinstance(x, ltype), layers))


def proc_dense_layer(layer, norm=False, epsilon=0):
    """Process a single layer if it is Dense (or other given type)."""
    assert isinstance(layer, Dense)
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
    if epsilon != 0:
        cond = compressed_dense < epsilon
        compressed_dense[cond] = 0
    if norm:
        compressed_dense *= norms_dense[:, np.newaxis]
    return compressed_dense, bias


@EX.capture
def evaluate(model, preproc_args, compile_args, gen_args, eval_args):
    """Evaluate model."""
    model.compile(**compile_args)
    gen_args.update(preproc_args)
    generator = CropGenerator(**gen_args)
    result = model.evaluate_generator(generator, **eval_args)
    return result


@EX.capture
def proc_model(model_name, proc_args):
    """
    Process one model based on the model name.  If proc_args is {} or
    None then evaluate all models, as provided by keras, otherwise
    process the Dense layers using some method.
    """
    # because of sacred:
    # pylint: disable=no-value-for-parameter
    model_dic = {"xception":
                 (Kapp.xception.Xception,
                  {'preproc': Kapp.xception.preprocess_input,
                   'target_size': 299}),
                 "vgg16":
                 (Kapp.vgg16.VGG16,
                  {'preproc': Kapp.vgg16.preprocess_input,
                   'target_size': 224}),
                 "vgg19":
                 (Kapp.vgg19.VGG19,
                  {'preproc': Kapp.vgg19.preprocess_input,
                   'target_size': 224}),
                 "resnet50":
                 (Kapp.resnet50.ResNet50,
                  {'preproc': Kapp.resnet50.preprocess_input,
                   'target_size': 224}),
                 "inceptionv3":
                 (Kapp.inception_v3.InceptionV3,
                  {'preproc': Kapp.inception_v3.preprocess_input,
                   'target_size': 299}),
                 "inceptionresnetv2":
                 (Kapp.inception_resnet_v2.InceptionResNetV2,
                  {'preproc': Kapp.inception_resnet_v2.preprocess_input,
                   'target_size': 299}),
                 "mobilenet":
                 (Kapp.mobilenet.MobileNet,
                  {'preproc': Kapp.mobilenet.preprocess_input,
                   'target_size': 224}),
                 "mobilenetv2":
                 (Kapp.mobilenet_v2.MobileNetV2,
                  {'preproc': Kapp.mobilenet_v2.preprocess_input,
                   'target_size': 224}),
                 "densenet121":
                 (Kapp.densenet.DenseNet121,
                  {'preproc': Kapp.densenet.preprocess_input,
                   'target_size': 224}),
                 "densenet169":
                 (Kapp.densenet.DenseNet169,
                  {'preproc': Kapp.densenet.preprocess_input,
                   'target_size': 224}),
                 "densenet201":
                 (Kapp.densenet.DenseNet201,
                  {'preproc': Kapp.densenet.preprocess_input,
                   'target_size': 224}),
                 "nasnetmobile":
                 (Kapp.nasnet.NASNetMobile,
                  {'preproc': Kapp.nasnet.preprocess_input,
                   'target_size': 224}),
                 "nasnetlarge":
                 (Kapp.nasnet.NASNetLarge,
                  {'preproc': Kapp.nasnet.preprocess_input,
                   'target_size': 331})}
    model_cls, preproc_args = model_dic[model_name]
    model = model_cls()
    layers = get_same_type_layers(model.layers)
    if not layers:
        # If the model has no dense layers, skip it by returning None.
        return None
    if not proc_args:
        # if proc_args == None or {} then just evaluate.
        return evaluate(model, preproc_args)
    for layer in layers:
        new_layer = proc_dense_layer(layer, **proc_args)
        layer.set_weights(new_layer)
    result = evaluate(model, preproc_args)
    return result


@EX.automain
def proc_all_models(gen_args, json_name):
    """Process all models."""
    model_names = ["xception", "vgg16", "vgg19", "resnet50", "inceptionv3",
                   "inceptionresnetv2", "mobilenet", "mobilenetv2",
                   "densenet121", "densenet169", "densenet201", "nasnetmobile",
                   "nasnetlarge"]

    if gen_args['fast_mode']:
        model_names = [model_names[3]]
    basedir = EX.observers[0].basedir
    result_file = os.path.join(basedir, "{}.json".format(json_name))
    aggregation = {}  # aggregate all results in a dictionary
    for index, name in enumerate(model_names):
        result = proc_model(name)
        # If proc model returned none, then it did nothing so skip.
        if result:
            print(">>>>>> {} - {}/{} Done.".format(name, index + 1,
                                                   len(model_names)))
            print(">>>>>> {} result = {}".format(name, result))
            aggregation[name] = result
    json.dump(aggregation, open(result_file, "w"))
    return aggregation
