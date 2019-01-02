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
    def __init__(self, val_file, img_dir, batch_size):
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


class BboxEvalGenerator(keras.utils.Sequence):
    """
    NOT FINISHED: not sure what to do with the bounding boxes, how to
    cut/corp.

    ResNet50 evaluation Sequence with bbox cutting for imagenet, might
    be ok for other models as well.
    """
    def _make_bbox_list(self):
        import xml.etree.ElementTree as ET
        bbox_template = ['xmin', 'ymin', 'xmax', 'ymax']
        self.bbox_list = []
        xml_list = os.listdir(self.xml_dir)
        for xml_name in xml_list:
            xml_path = os.path.join(self.xml_dir, xml_name)
            root = ET.parse(xml_path)
            filename = root.find('filename').text
            img_path = os.path.join(self.img_dir, filename + ".JPEG")
            objs = root.findall('object')
            for obj in objs:
                bbox = map(obj.find('bndbox').find, bbox_template)
                bbox = list(map(lambda t: int(t.text), bbox))
                wnid = obj.find('name').text
                wnid = self.wnid_list.index(wnid)
                self.bbox_list.append([img_path, tuple(bbox), wnid])

    def __init__(self, img_dir, xml_dir, wnid_path, batch_size):
        self.wnid_list = open(wnid_path).read().splitlines()
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self._make_bbox_list()
        self.indices = np.arange(len(self.bbox_list))
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.bbox_list) / self.batch_size)

    def __getitem__(self, idx):
        print('Evaluating idx: {}/{}'.format(idx, self.__len__()))
        inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        inputs_batch = np.zeros([self.batch_size, 224, 224, 3], np.float32)
        outputs_batch = np.zeros([self.batch_size, 1000], np.float32)
        for i, j in enumerate(inds):
            img_path, bbox, wnid_idx = self.bbox_list[j]
            xmin, ymin, xmax, ymax = bbox
            img = image.load_img(img_path)
            height, width, _ = img.shape
            crop_height, crop_width = ymax - ymin, xmax - xmin
            crop_max_dim = max(crop_height, crop_width)

            # new_height = height * 256 // min(img.shape[:2])
            # new_width = width * 256 // min(img.shape[:2])
            # img = img.resize(new_width, new_height)

            img = img.crop(bbox)
            img.show()
            img = image.img_to_array(img)
            img = preprocess_input(img)
            inputs_batch[i] = img
            outputs_batch[i] = keras.utils.to_categorical(cat, 1000)
        return inputs_batch, outputs_batch


if __name__ == '__main__':
    PARAM = {
        'wnid_path': "/home/vatai/tmp/ilsvrc/caffe_ilsvrc12/synsets.txt",
        'img_dir': "/home/vatai/tmp/ilsvrc/db",
        'xml_dir': "/home/vatai/tmp/ilsvrc/val",
        'batch_size': 2
    }
    bbox_test = False
    if bbox_test:
        gen = BboxEvalGenerator(**PARAM)
        import matplotlib.pyplot as plt
        x, y = gen[0]
        plt.imshow(x[0])
        plt.show()
        print(np.argmax(y))
    alt_test = True
