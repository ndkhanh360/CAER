import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch 

class CNN2DBlock(nn.Module):
    def __init__(self, conv_num, cnn_num_kernels, cnn_kernel_size=3, bn=True, relu=True, maxpool=True, maxpool_kernel_size=2):
        """
        A CNN2DBlock with architecture (CNN + BN + ReLU + max-pool) x N
        Inputs:
        - conv_num (int): number of convolution layers
        - cnn_num_kernels (list of int with size conv_num + 1): the first elements is the number of channel of the inputs, the rest are theh number of kernels in each convolution layer
        - cnn_kernel_size (int): size of kernels
        - bn (boolean): use batch normalization or not
        - relu (boolean): use relu or not
        - maxpool (boolean): use maxpool layers or not (number of maxpool layers = conv_num - 1)
        - maxpool_kernel_size (int): size of maxpool kernel

        """
        super().__init__()
        padding = int((cnn_kernel_size - 1)/2)
        self.convs = nn.ModuleList([nn.Conv2d(cnn_num_kernels[i], cnn_num_kernels[i+1], kernel_size=cnn_kernel_size, padding=padding) for i in range(conv_num)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(cnn_num_kernels[i+1]) for i in range(conv_num)]) if bn else None
        self.relu = nn.ReLU() if relu else None 
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size) if maxpool else None

    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            if self.relu is not None:
                x = self.relu(x)
            if i < n-1 and self.maxpool is not None:
                x = self.maxpool(x)

        return x 

class TwoStreamNetwork(nn.Module):
    def __init__(self, context=True, attention=True):
        super().__init__()
        self.face_encoding_module = CNN2DBlock(conv_num=5, cnn_num_kernels=[3, 32, 64, 128, 256, 256])
        self.context_encoding_module = CNN2DBlock(conv_num=5, cnn_num_kernels=[3, 32, 64, 128, 256, 256]) if context else None
        self.attention_inference_module = CNN2DBlock(conv_num=2, cnn_num_kernels=[256, 128, 1], maxpool=False) if attention else None

    def forward(self, face, context=None):
        face = self.face_encoding_module(face)
        if context is not None:
            context = self.context_encoding_module(context)
            if self.attention_inference_module is not None:
                attention = self.attention_inference_module(context)
                context = context * attention      
        face_size, context_size = face.shape[2], context.shape[2]
        face = F.avg_pool2d(face, kernel_size=face_size)
        context = F.avg_pool2d(context, kernel_size=context_size)

        return face, context

class FusionNetwork(nn.Module):
    def __init__(self, num_class=7, attention=True):
        super().__init__()
        self.face_stream_conv = CNN2DBlock(conv_num=2, cnn_num_kernels=[256, 128, 1], cnn_kernel_size=1, bn=False, relu=False, maxpool=False) if attention else None
        self.context_stream_conv = CNN2DBlock(conv_num=2, cnn_num_kernels=[256, 128, 1], cnn_kernel_size=1, bn=False, relu=False, maxpool=False) if attention else None

        self.conv1 = nn.Conv2d(256*2, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, num_class, kernel_size=1)
        self.dropout = nn.Dropout2d()

    def forward(self, face, context):
        if self.face_stream_conv is not None and self.context_stream_conv is not None:
            face_weights = self.face_stream_conv(face)
            context_weights = self.context_stream_conv(context)

            weights = torch.cat([face_weights, context_weights], dim=1)
            weights = F.softmax(weights, dim=1)

            face = face * weights[:, 0, :].unsqueeze(dim=1)
            context = context * weights[:, 1, :].unsqueeze(dim=1)

        features = torch.cat([face, context], dim=1)
        features = F.relu(self.conv1(features))
        features = self.dropout(features)
        features = self.conv2(features)

        return features

class CAERSNet(BaseModel):
    def __init__(self, context=True, context_attention=True, fusion_attention=True):
        super().__init__()
        self.two_stream_net = TwoStreamNetwork(context=context, attention=context_attention)
        self.fusion_net = FusionNetwork(attention=fusion_attention)

    def forward(self, face, context=None):
        face, context = self.two_stream_net(face, context)
        features = self.fusion_net(face, context)
        N, K = features.shape[:2]
        features = features.view(N, K)

        return features

