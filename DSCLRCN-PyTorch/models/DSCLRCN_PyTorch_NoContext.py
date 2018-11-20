"""LocalFeaturesCNN"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_vgg16.local_cnn import LocalFeatsCNN
from models.places_vgg16.places_cnn import PlacesCNN
from models.segmentation_resnet50.segmentation_nn import SegmentationNN

import numpy as np


class DSCLRCN(nn.Module):

    def __init__(self, input_dim=(96, 128), LSTMs_input_size=(128*12, 128*16), local_feats_net='CNN'):#, LSTM_hs=256):
        super(DSCLRCN, self).__init__()

        self.input_dim = input_dim
        self.LSTMs_isz = LSTMs_input_size

        if local_feats_net == 'Seg':
            self.local_feats = SegmentationNN()
        else:
            self.local_feats = LocalFeatsCNN()


        #print('Constructing LSTMs')
        self.lstm_h = nn.LSTM(LSTMs_input_size[0], LSTMs_input_size[0], 1, batch_first=True)
        self.lstm_v = nn.LSTM(LSTMs_input_size[1], LSTMs_input_size[1], 1, batch_first=True)

        #print('conv_last')
        self.last_conv = nn.Conv2d(128, 1, 1)

        #print('softmax & upsample')
        
        self.upsample = nn.Upsample(size=input_dim, mode='bilinear')
        
        self.score = nn.Softmax(dim=2)


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        #print('Start FP')
        N = x.size(0)
        H,W = self.input_dim
        
        local_feats = self.local_feats(x)
        H_lf, W_lf = local_feats.size()[2:]
        

        local_feats_h = local_feats.contiguous().view(N, W_lf, self.LSTMs_isz[0])

        output_h1, hz1 = self.lstm_h(local_feats_h)

        #print('2nd', output_h1.size())
        output_h1 = output_h1.contiguous().view(N, 128, H_lf, W_lf)
        
        output_h12 = output_h1
        
        output_h12v = output_h12.contiguous().view(N, H_lf, self.LSTMs_isz[1])
        
        #print('3rd LSTM')
        output_h12v1, hz3 = self.lstm_v(output_h12v)
        output_h12v1 = output_h12v1.contiguous().view(N, 128, H_lf, W_lf)
        
        
        output_h12v12 = output_h12v1
        
        output_conv = self.last_conv(output_h12v12)
        
        #print(output_conv.size())
        N, C, H_l, W_l, = output_conv.size()
        
        #print('Upsampling')
                        
        output_upsampled = self.upsample(output_conv)
        
        output_score = self.score(output_upsampled.contiguous().view(N, C, -1))
        
        output_score = output_score.contiguous().view(N, C, H, W)

        return output_score

    
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
