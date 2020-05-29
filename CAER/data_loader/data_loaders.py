from torchvision import datasets, transforms
from base import BaseDataLoader
import torch
from torch.utils.data import DataLoader
import utils.util as ut 

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class DataLoaderGenerator(object):
    def __init__(self, root, test_size=0, batch_size=32, num_workers=2, mode=0):
        self.test_size = test_size
        if test_size > 0:
            (train_paths, val_paths), self.class_to_idx = ut.get_path_images(root, test_size, mode)
            train_transfrom, val_transform = ut.get_transform(), ut.get_transform(train=False)
            self.train_dset, self.val_dset = ut.MyDataset(train_paths, train_transfrom), ut.MyDataset(val_paths, val_transform)
            self.train_loader = DataLoader(self.train_dset, batch_size=batch_size, 
                                shuffle=True, collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
            self.val_loader = DataLoader(self.val_dset, batch_size=batch_size, 
                                collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
        else:
            test_paths, self.class_to_idx = ut.get_path_images(root, test_size, mode)
            test_transform = ut.get_transform(train=False)
            self.test_dset = ut.MyDataset(test_paths, test_transform)
            self.test_loader = DataLoader(self.test_dset, batch_size=batch_size, 
                                collate_fn=collate_fn, num_workers=num_workers, drop_last=True)

    def get_loaders(self):
        if self.test_size > 0:
            return (self.train_loader, self.val_loader), self.class_to_idx
        return self.test_loader, self.class_to_idx

