import json
import numpy as np 
import pandas as pd
from pathlib import Path
from PIL import Image
from itertools import repeat
from collections import OrderedDict
import torchvision 
from torchvision import transforms
import torchvision.datasets as dset
import cv2
import torch 
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import dlib 

def get_path_images(root, test_size=0.0, mode=0):
    """
    Get image paths to create dataset
    Args:
    - root: image folder
    - test_size: validation ratio 
    - mode: 0 (full dataset), n (1024*n images)
    """
    img_folder = ImageFolder(root)
    train = img_folder.imgs
    np.random.seed(42)
    np.random.shuffle(train)

    if mode != 0: # debugging
        train = train[:1024*mode]
    class_to_idx = img_folder.class_to_idx

    if test_size > 0.0:
        num_train = int((1-test_size)*len(train))
        print('Trainset size: {}, Valset size: {}'.format(num_train, len(train)-num_train))
        return (train[:num_train], train[num_train:]), class_to_idx
    print('Testset size:', len(train))
    return train, class_to_idx

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path)

        if self.transform is not None:
            try:
                image = self.transform(image)
            except:
                return None
        return image, label

class ExtractFaceContext(object):
    """
    Extract facial region and context from an image.

    Args:
    detector: face detector
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, image):
        np_img = np.asarray(image)
        bb = self.detector(np_img, 1)[0]
        face, context = np_img.copy(), np_img.copy()
        x1, y1, x2, y2 = bb.left(), bb.top(), bb.right(), bb.bottom()
        face = face[y1:y2, x1:x2]
        cv2.rectangle(context, (x1,y1), (x2, y2), (0,0,0), -1)

        return {
            'face': Image.fromarray(face),
            'context': Image.fromarray(context)
        }

class ResizeFaceContext(object):
    """
    Resize facial region and context

    Args:
    sizes (tuple): size of facial region and context region
    """

    def __init__(self, sizes):
        assert isinstance(sizes, tuple)
        self.sizes = sizes

    def __call__(self, sample):
        face_size, context_size = self.sizes
        if isinstance(face_size, int):
            face_size = (face_size, face_size)
        if isinstance(context_size, int):
            context_size = (context_size, context_size)

        face, context = sample['face'], sample['context']
        return {
            'face': transforms.Resize(face_size)(face),
            'context': transforms.Resize(context_size)(context)
        }

class Crop(object):
    """
    (Randomly) crop context region

    Args:
    size (int): context region size
    mode (string): takes value "train" or "test". If "train", use random crop.
                    If "test", use center crop.
    """

    def __init__(self, size, mode="train"):
        self.size = size
        self.mode = mode

    def __call__(self, sample):
        context = sample['context']
        if self.mode == "train":
            context = transforms.RandomCrop(self.size)(context)
        else:
            context = transforms.CenterCrop(self.size)(context)

        return {
            'face': sample['face'],
            'context': context
        }

class ToTensorAndNormalize(object):
    """
    Convert PIL image to Tensor
    """
    def __call__(self, sample):
        face, context = sample['face'].convert("RGB"), sample['context'].convert("RGB")
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        return {
            'face': normalize(toTensor(face)),
            'context': normalize(toTensor(context)),
        }

def get_transform(train=True):
  return transforms.Compose([transforms.Resize(224),
                            ExtractFaceContext(), # edit this line
                            ResizeFaceContext((96, (128, 171))),
                            (Crop(112, "train") if train else Crop(112, "test")),
                             ToTensorAndNormalize()])

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
