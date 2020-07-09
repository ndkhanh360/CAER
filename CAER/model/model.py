import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
from torchvision.models import resnet18
import torch

class Encoder(nn.Module):
    def __init__(self, num_kernels, kernel_size=3, bn=True, max_pool=True, maxpool_kernel_size=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        n = len(num_kernels) - 1
        self.convs = nn.ModuleList([nn.Conv2d(num_kernels[i], num_kernels[i+1], kernel_size, padding=padding) for i in range(n)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_kernels[i+1]) for i in range(n)]) if bn else None
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size) if max_pool else None
    
    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x)
            if self.max_pool is not None: # check if i < n
                x = self.max_pool(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class TwoStreamNetwork(nn.Module):
    def __init__(self, fine_tune=True, path=None):
        super().__init__()
        assert path is not None 
        num_kernels = [3, 32, 64, 128, 256, 512]

        print('Loading checkpoints from pretrained...')
        checkpoint = torch.load(path)
        self.face_encoding_module = resnet18()

        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = self.face_encoding_module.state_dict()
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                new_key = key[7:]
                if new_key in model_state_dict.keys():
                    model_state_dict[new_key] = pretrained_state_dict[key]
        self.face_encoding_module.load_state_dict(model_state_dict)  
        self.face_encoding_module.fc = Identity()

        for param in self.face_encoding_module.parameters():
            param.requires_grad = fine_tune

        self.context_encoding_module = Encoder(num_kernels)
        self.context_attention_inference_module = Encoder([512, 64, 1], max_pool=False)
    
    def forward(self, face, context):
        face = self.face_encoding_module(face) # N, 512

        context = self.context_encoding_module(context)
        attention = self.context_attention_inference_module(context)
        N, C, H, W = attention.shape
        attention = F.softmax(attention.reshape(N, C, -1), dim=2).reshape(N, C, H, W)
        context = context * attention 

        return face, context

class FusionNetwork(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()
        self.face_1 = nn.Linear(512, 128)
        self.face_2 = nn.Linear(128, 1)

        self.context_1 = nn.Linear(512, 128)
        self.context_2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(512*2, 128)
        self.fc2 = nn.Linear(128, num_class)

        self.dropout = nn.Dropout()
    
    def forward(self, face, context):
        # face = F.avg_pool2d(face, face.shape[2]).reshape(face.shape[0], -1)
        context = F.avg_pool2d(context, context.shape[2]).reshape(context.shape[0], -1)
        # face = face.reshape(face.shape[0], -1)
        # context = context.reshape(context.shape[0], -1)

        lambda_f = F.relu(self.face_1(face))
        lambda_c = F.relu(self.context_1(context))

        lambda_f = self.face_2(lambda_f)
        lambda_c = self.context_2(lambda_c)

        weights = torch.cat([lambda_f, lambda_c], dim=-1)
        weights = F.softmax(weights, dim=1)

        face = face *  weights[:, 0].unsqueeze(dim=-1)
        context = context * weights[:, 1].unsqueeze(dim=-1)
        
        features = torch.cat([face, context], dim=-1)

        features = F.relu(self.fc1(features))

        features = self.dropout(features)

        return self.fc2(features)

class CAERSNet(BaseModel):
    def __init__(self, fine_tune=True, path=None):
        super().__init__()
        assert path is not None 
        self.two_stream_net = TwoStreamNetwork(fine_tune, path)
        self.fusion_net = FusionNetwork()
          
    def forward(self, face=None, context=None):
        face, context = self.two_stream_net(face, context)

        return self.fusion_net(face, context)
