"""Custom Keras generators."""

import math
import os
import numpy as np
import keras.utils
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input


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
