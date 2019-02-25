"""
First attempt to write a program to evaluate an image classification
network.

Probable structure.

import network

maybe def transform

def loader

score, total = 0, 0
while loader:
  y = net(x)
  score += count(y, ytrue)
  total += batch_size

return score/total
"""

import os
from skimage import io
import numpy as np
import sacred
import torch
import torchvision


class ImageNet(torch.utils.data.Dataset):
    """Load batches from imagenet."""
    def __init__(self, img_dir, val_file, transform=None):
        self.file_list = []
        self.category_list = []
        with open(val_file) as file:
            for line in file:
                img_path, cat_str = line.split(" ")
                cat = int(cat_str)
                img_path = os.path.join(img_dir, img_path)
                self.file_list.append(img_path)
                self.category_list.append(cat)

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = io.imread(self.file_list[idx])
        sample = {'image': image, 'category': self.category_list[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, categoyr = sample['image'], sample['category']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'category': torch.zeros(1000, dtype=torch.long)}


EX = sacred.Experiment()
# EX.captured_out_filter = apply_backspaces_and_linefeeds
#EX.observers.append(FileStorageObserver.create(get_results_dir(__file__)))
#EX.observers.append(TelegramObserver.from_config('telegram.json'))


@EX.config
def config():
    """Config function for the experiment."""
    testset_args = {'img_dir': os.path.expanduser("~/tmp/ilsvrc/db"),
                    'val_file': os.path.expanduser("~/tmp/ilsvrc/caffe_ilsvrc12/val.txt")}


@EX.automain
def main(testset_args):
    net = torchvision.models.resnet50(pretrained=True)
    testset = ImageNet(#transform=ToTensor(),
                       **testset_args)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    a = next(iter(testloader))
    print(a['image'].size())
    with torch.no_grad():
        total, correct = 0, 0
        for data in testloader:
            image, cats = data
            #print('WTF', data)
            y = net(data['image'])
            _, pred_cats = torch.max(y.data, 1)
            total += 4
            # correct += (pred_cats == cats).sum().item()
            # print(i, np.shape(data['image']))
            return
