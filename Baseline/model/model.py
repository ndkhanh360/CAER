import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel
from torchvision import models

class ResNet(BaseModel):
    def __init__(self, drop_out=False, num_classes=7, fine_tune=True, path=None):
        super().__init__()
        if path is not None:
            print('Loading checkpoints from pretrained...')
            checkpoint = torch.load(path)
            self.model = models.resnet18()

            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = self.model.state_dict()
            for key in pretrained_state_dict:
                if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                    pass
                else:    
                    new_key = key[7:]
                    if new_key in model_state_dict.keys():
                        model_state_dict[new_key] = pretrained_state_dict[key]
            self.model.load_state_dict(model_state_dict)  
        else:
            self.model = models.resnet152(pretrained=True)
          

        for param in self.model.parameters():
            param.requires_grad = fine_tune ## fix this 
                
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        ) if drop_out else nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class AlexNet(BaseModel):
    def __init__(self, drop_out=False, num_classes=7, fine_tune=True):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False 

        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        ) if drop_out else nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class VGGNet(BaseModel):
    def __init__(self, drop_out=False, num_classes=7, fine_tune=True):
        super().__init__()
        self.model = models.vgg19(pretrained=True)

        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        ) if drop_out else nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DumbNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(224*224*3, 7)
    
    def forward(self, x):
        return self.fc(x.reshape(-1, 224*224*3))