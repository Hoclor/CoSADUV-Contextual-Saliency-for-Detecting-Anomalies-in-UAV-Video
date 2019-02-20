"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_resnet50.models import ModelBuilder

# L2Norm layer code from https://github.com/clcarwin/SFD_pytorch/blob/master/net_s3fd.py
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale=1.0):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = x / norm * self.weight.view(1,-1,1,1)
        return x

class SegmentationNN(nn.Module):

    def __init__(self, models_path='models/segmentation_resnet50/'):
        super(SegmentationNN, self).__init__()
        
        encoder_path = models_path+'resnet50_kaggle.pth'
        decoder_path = models_path+'decoder_best.pth'
        
        builder = ModelBuilder()
        self.net_encoder = builder.build_encoder(weights=encoder_path)
        
        # conv6 - reduce dimensionality from 2048 to 512
        self.conv6 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, dilation=1)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.l2norm = L2Norm(512, scale=400)
        
    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        # Apply conv1 to conv5
        x = self.net_encoder(x)
        
        # conv6 - reduce dimensionality from 2048 to 512
        x = self.relu6(self.conv6(x))
        
        x = self.l2norm(x)
        
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
