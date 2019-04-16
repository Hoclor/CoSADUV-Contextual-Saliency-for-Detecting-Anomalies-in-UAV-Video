"""SegmentationNN"""
import torch
import torch.nn as nn
from models.places_vgg16 import PlacesCNN as PCNN
import math

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
        x = x / norm * self.weight.view(1,-1)
        return x

class PlacesCNN(nn.Module):

    def __init__(self, model_path='models/places_vgg16/PlacesCNN.pth', input_dim=(96, 128)):
        super(PlacesCNN, self).__init__()

        complete_model = PCNN.PlacesCNN
        
        self.feats = nn.Sequential(*list(complete_model.children())[0:31])
        model_dict = self.feats.state_dict()
        
        print('Loading weights for PlacesCNN_VGG16')
        pretrained_dict = torch.load(model_path)
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        self.feats.load_state_dict(pretrained_dict)
        
        self.fc = nn.Linear(512*math.ceil(input_dim[0]/32)*math.ceil(input_dim[1]/32), 128) # (512*3*4, 128)
        
        self.l2norm = L2Norm(128, scale=9.0)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        x = self.relu(self.feats(x))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        x = self.relu(x)
        
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
