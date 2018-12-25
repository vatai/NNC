"""Keras pretrained model evaluation prototype."""

import os.path
import math
import numpy as np
import keras.utils
from keras.preprocessing import image
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
import sacred

EX = sacred.Experiment(interactive=True, name='evaluation test')


@EX.config
def config():
    # flake8: noqa: F841
    # pylint: disable=unused-variable
    compile_args = {'optimizer': 'RMSprop',
                    'loss': 'categorical_crossentropy',
                    'metrics': [categorical_accuracy,
                                top_k_categorical_accuracy]}
    gen_args = {'db_path': "/home/vatai/tmp/ilsvrc/db",
                'val_file': "/home/vatai/tmp/ilsvrc/caffe_ilsvrc12/val.txt",
                'batch_size': 64}
    eval_args = {'max_queue_size': 10,
                 'workers': 1,
                 'use_multiprocessing': False}


class EvalGenerator(keras.utils.Sequence):
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
        print('Evaluating idx: {}/{}'.format(idx, self.__len__()))
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


@EX.automain
def main(compile_args, gen_args, eval_args):
    model = ResNet50(weights='imagenet')
    model.compile(**compile_args)
    generator = EvalGenerator(**gen_args)
    result = model.evaluate_generator(generator, **eval_args)
    # print('generator len', len(generator))
    print('result', result)
