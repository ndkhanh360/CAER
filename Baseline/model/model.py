import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models

class ResNet(BaseModel):
    def __init__(self, drop_out=False, num_classes=7, fine_tune=True):
        super().__init__()
        self.model = models.resnet152(pretrained=True)

        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False 
                
        num_features = self.model.fc.in_features
        self.fc = nn.Sequential(
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