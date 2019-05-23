"""Custom Keras generators."""

import math
import os
import numpy as np
import keras.utils
from keras.preprocessing import image


class CropGenerator(keras.utils.Sequence):
    """Evaluation generator class (Sequence) for ImageNet 2012 with
    cropping.

    """
    def __init__(self, batch_size,
                 preproc,
                 load_size=256, target_size=224,
                 img_dir=os.path.expanduser("~/tmp/ilsvrc/db"),
                 val_file=os.path.expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt"),
                 fast_mode=0):
        self.img_dir = img_dir
        self.file_list = []
        self.category_list = []
        with open(val_file) as file:
            for line in file:
                img_path, cat_str = line.split(" ")
                cat = int(cat_str)
                img_path = os.path.join(img_dir, img_path)
                self.file_list.append(img_path)
                self.category_list.append(cat)
        self.batch_size = batch_size
        self.preproc = preproc
        self.load_size = max(load_size, target_size)
        self.target_size = target_size
        self.indices = np.arange(len(self.file_list))
        self.fast_mode = fast_mode

    def __len__(self):
        if self.fast_mode:
            return self.fast_mode if isinstance(self.fast_mode, int) else 1
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        input_shape = [self.batch_size, self.target_size, self.target_size, 3]
        inputs_batch = np.zeros(input_shape, np.float32)
        outputs_batch = np.zeros([self.batch_size, 1000], np.float32)
        for i, j in enumerate(inds):
            load_size = (self.load_size, self.load_size)
            img = image.load_img(self.file_list[j], target_size=load_size)
            # 256 - 224 = 32
            offset = (self.load_size - self.target_size)
            box = [offset, offset,
                   self.target_size + offset, self.target_size + offset]
            if self.load_size != self.target_size:
                img = img.crop(box)
            img = image.img_to_array(img)
            img = self.preproc(img)
            inputs_batch[i] = img
            cat = self.category_list[j]
            outputs_batch[i] = keras.utils.to_categorical(cat, 1000)
        return inputs_batch, outputs_batch
