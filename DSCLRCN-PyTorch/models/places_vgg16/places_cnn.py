"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.places_vgg16 import PlacesCNN as PCNN


class PlacesCNN(nn.Module):

    def __init__(self, model_path='models/places_vgg16/PlacesCNN.pth'):
        super(PlacesCNN, self).__init__()

        complete_model = PCNN.PlacesCNN
        
        self.feats = nn.Sequential(*list(complete_model.children())[0:31])
        model_dict = self.feats.state_dict()
        
        print('Loading weights for PlacesCNN_VGG16')
        pretrained_dict = torch.load(model_path)
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        self.feats.load_state_dict(pretrained_dict)
        
        self.fc = nn.Linear(512*3*4, 128)
        
        #self.upsample = nn.Upsample(size=output_dim, mode='bilinear')
                        

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        x = F.relu(self.feats(x))
        
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        #x = self.upsample()
        
        norm = x.norm(p=2, dim=1, keepdim=True)
        x_norm = x.div(norm.expand_as(x) + 1e-8)*100

        return x_norm

    
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
