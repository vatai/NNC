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
            outputs_batch[i] = keras.utils.to_categorical(wnid_idx, 1000)
        return inputs_batch, outputs_batch


class CropGenerator(keras.utils.Sequence):
    """
    Evaluation generator class (Sequence) for ImageNet 2012 with
    croping.  For now it is ResNet50 specific.
    """
    def __init__(self, val_file, img_dir, batch_size,
                 preproc=preprocess_input,
                 load_size=256, target_size=224):
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

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, idx):
        print('Evaluating idx: {}/{}'.format(idx, self.__len__()))
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
