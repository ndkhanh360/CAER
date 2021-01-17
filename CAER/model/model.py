import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch

class Encoder(nn.Module):
    def __init__(self, num_kernels, kernel_size=3, bn=True, max_pool=True, maxpool_kernel_size=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        n = len(num_kernels) - 1
        self.convs = nn.ModuleList([nn.Conv2d(
            num_kernels[i], num_kernels[i+1], kernel_size, padding=padding) for i in range(n)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_kernels[i+1])
                                 for i in range(n)]) if bn else None
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size) if max_pool else None

    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x) 
            if self.max_pool is not None and i < n-1:  # check if i < n
                x = self.max_pool(x)
        return x


class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_kernels = [3, 32, 64, 128, 256, 256]
        self.face_encoding_module = Encoder(num_kernels)
        self.context_encoding_module = Encoder(num_kernels)
        self.attention_inference_module = Encoder(
            [256, 128, 1], max_pool=False)

    def forward(self, face, context):
        face = self.face_encoding_module(face)

        context = self.context_encoding_module(context)
        attention = self.attention_inference_module(context)
        N, C, H, W = attention.shape
        attention = F.softmax(attention.view(
            N, -1), dim=-1).view(N, C, H, W)
        context = context * attention

        return face, context


class FusionNetwork(nn.Module):
    def __init__(self, use_face=True, use_context=True, concat=False, num_class=7):
        super().__init__()
        # add batch norm to ensure the mean and std of 
        # face and context features are not too different
        self.face_bn = nn.BatchNorm1d(256)
        self.context_bn = nn.BatchNorm1d(256)

        self.use_face, self.use_context = use_face, use_context
        self.concat = concat

        self.face_1 = nn.Linear(256, 128)
        self.face_2 = nn.Linear(128, 1)

        self.context_1 = nn.Linear(256, 128)
        self.context_2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_class)

        self.dropout = nn.Dropout()

    def forward(self, face, context):
        face = F.avg_pool2d(face, face.shape[2]).view(face.shape[0], -1)
        context = F.avg_pool2d(context, context.shape[2]).view(context.shape[0], -1)

        # add batch norm for face and context branch
        face, context = self.face_bn(face), self.context_bn(context)

        if not self.concat:
            lambda_f = F.relu(self.face_1(face))
            lambda_c = F.relu(self.context_1(context))

            lambda_f = self.face_2(lambda_f)
            lambda_c = self.context_2(lambda_c)

            weights = torch.cat([lambda_f, lambda_c], dim=-1)
            weights = F.softmax(weights, dim=-1)
            face = face * weights[:, 0].unsqueeze(dim=-1)
            context = context * weights[:, 1].unsqueeze(dim=-1)

        if not self.use_face:
            face = torch.zeros_like(face)

        if not self.use_context:
            context = torch.zeros_like(context)

        features = torch.cat([face, context], dim=-1)
        features = F.relu(self.fc1(features))
        features = self.dropout(features)

        return self.fc2(features)


class CAERSNet(BaseModel):
    def __init__(self, use_face=True, use_context=True, concat=False):
        super().__init__()
        self.two_stream_net = TwoStreamNetwork()
        self.fusion_net = FusionNetwork(use_face, use_context, concat)

    def forward(self, face=None, context=None):
        face, context = self.two_stream_net(face, context)

        return self.fusion_net(face, context)
