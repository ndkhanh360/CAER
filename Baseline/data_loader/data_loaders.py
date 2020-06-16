from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from base import BaseDataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)
    
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
        try:
            sample = self.data[idx].split(',')
            path, label, x1, y1, x2, y2 = os.path.join(self.root, sample[0]), int(sample[1]), int(sample[2]), int(sample[3]), int(sample[4]), int(sample[5])
            image = Image.open(path).crop((x1, y1, x2, y2))
            if self.transform is not None:
                image = self.transform(image)
        except:
            return None
        return image, label

class CAERSDataLoader(BaseDataLoader):
    def __init__(self, root, detect_file=None, batch_size=32, shuffle=True, validation_split=0.0, num_workers=2):
        """
        Create dataloader from directory
        Args:
            - root (str): root directory
            - detect_file (str): file containing results from detector (default None). If detect_file is specified, 
                                 the input to the model is the face region of each image, otherwise it will be the whole image.
            - validation_split (int or float): the number of images/the ratio of the dataset used for validation
        """
        
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # if detect_file is not None:
        #     self.dataset, self.collate_function = MyDataset(root, detect_file, data_transforms), collate_fn
        # else:
        #     self.dataset, self.collate_function = ImageFolder(root, data_transforms), default_collate
        self.dataset = MyDataset(root, detect_file, data_transforms) if detect_file is not None else ImageFolder(root, data_transforms)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)