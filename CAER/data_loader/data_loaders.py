from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import DataLoader, Dataset
import utils.util as ut 
from PIL import Image, ImageDraw
import os

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class MyDataset(Dataset):
    def __init__(self, root, input_file, transform=None):
        self.root = root 
        self.transform = transform
        self.data = self.read_input_file(input_file)

    def read_input_file(self, file):
        return [line.rstrip('\n') for line in open(file)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError('Index out of bound')
        sample = self.data[idx].split(',')
        path, label, x1, y1, x2, y2 = os.path.join(self.root, sample[0]), int(sample[1]), int(sample[2]), int(sample[3]), int(sample[4]), int(sample[5])
        im = Image.open(path)

        # extract facial and context regions
        face = im.crop((x1, y1, x2, y2))
        draw = ImageDraw.Draw(im)
        draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0))
        data = {
            'face': face,
            'context': im
        }
        
        # transform data 
        try:
            if self.transform is not None:
                data = self.transform(data)
        except:
            return None

        return data, label

class CAERSDataLoader(BaseDataLoader):
    def __init__(self, root, detect_file, train=True, batch_size=32, shuffle=True, num_workers=2):
        """
        Create dataloader from directory
        Args:
            - root (str): root directory
            - detect_file (str): file containing results from detector 
        """
        
        data_transforms = ut.get_transform(train)
        self.dataset = MyDataset(root, detect_file, data_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split=0.0, num_workers=num_workers, collate_fn=collate_fn)
