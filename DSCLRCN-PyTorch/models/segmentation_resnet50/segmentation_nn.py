"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_resnet50.models import ModelBuilder


class SegmentationNN(nn.Module):

    def __init__(self, models_path='models/segmentation_resnet50/'):
        super(SegmentationNN, self).__init__()

        encoder_path = models_path+'encoder_best.pth'
        decoder_path = models_path+'decoder_best.pth'

        builder = ModelBuilder()
        self.net_encoder = builder.build_encoder(arch='resnet50_dilated8', fc_dim=512, weights=encoder_path)
        
        self.trans_feats1 = nn.Conv2d(2048, 512, 1)
        
        self.trans_feats2 = nn.Conv2d(512, 128, 1)

        #self.net_encoder.eval()
        #self.net_decoder = builder.build_decoder(arch='psp_bilinear', fc_dim=2048, weights=decoder_path)
        #self.net_decoder.eval()
        

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        x = self.net_encoder(x)
        #print(x.size())
        #x = self.net_decoder(x, segSize=(x.size(-2), x.size(-1)))
        
        x = F.relu(self.trans_feats1(x))
        
        x = self.trans_feats2(x)
        
        xVector = x.view(x.size(0), x.size(1), -1, 1)
        norm = xVector.norm(p=2, dim=2, keepdim=True)
        
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
