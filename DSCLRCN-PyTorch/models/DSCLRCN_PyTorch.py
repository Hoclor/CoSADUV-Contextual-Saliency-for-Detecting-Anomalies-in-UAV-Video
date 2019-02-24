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

    def __init__(self, input_dim=(96, 128), local_feats_net='CNN'):
        super(DSCLRCN, self).__init__()

        self.input_dim = input_dim
        
        # Input size of the LSTMs
        # LSTM_1 input size: channel * height, of local_feats output (i.e. 512 * input_height/8)
        # LSTM_2 input size: 256 * width (2 * 128 as LSTMs output 128 values, *2 for bidirectional LSTMs)
        # LSTM_3 input size: 256 * height (same reason as above)
        # LSTM_4 input size: 256 * width (same reason as above)
        self.LSTMs_isz = (512*input_dim[0]//8,
                          256*input_dim[1]//8,
                          256*input_dim[0]//8,
                          256*input_dim[1]//8)
        
        # Hidden size of the LSTMs
        # LSTM_1 hidden size: 128 * height (of local_feats output)
        # LSTM_2 hidden size: 128 * width (of local_feats output)
        # LSTM_3 hidden size: 128 * height (of local_feats output)
        # LSTM_4 hidden size: 128 * width (of local_feats output)
        self.LSTMs_hsz = (128*input_dim[0]//8,
                          128*input_dim[1]//8,
                          128*input_dim[0]//8,
                          128*input_dim[1]//8)

        if local_feats_net == 'Seg':
            self.local_feats = SegmentationNN()
        else:
            self.local_feats = LocalFeatsCNN()

        self.context = PlacesCNN(input_dim=input_dim)

        self.fc_h = nn.Linear(128, self.LSTMs_isz[0])
        self.fc_v = nn.Linear(128, self.LSTMs_isz[1])

        # Constructing LSTMs:
        self.blstm_h_1 = nn.LSTM(input_size=self.LSTMs_isz[0], hidden_size=self.LSTMs_hsz[0], num_layers=1, batch_first=True, bidirectional=True)
        self.blstm_v_1 = nn.LSTM(input_size=self.LSTMs_isz[1], hidden_size=self.LSTMs_hsz[1], num_layers=1, batch_first=True, bidirectional=True)
        self.blstm_h_2 = nn.LSTM(input_size=self.LSTMs_isz[2], hidden_size=self.LSTMs_hsz[2], num_layers=1, batch_first=True, bidirectional=True)
        self.blstm_v_2 = nn.LSTM(input_size=self.LSTMs_isz[3], hidden_size=self.LSTMs_hsz[3], num_layers=1, batch_first=True, bidirectional=True)

        # Last conv to move to one channel
        self.last_conv = nn.Conv2d(2*128, 1, 1)

        # softmax
        self.score = nn.Softmax(dim=2)


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        N = x.size(0)
        H,W = self.input_dim
        
        # Get local feature map
        local_feats = self.local_feats(x)
        H_lf, W_lf = local_feats.size()[2:]

        # Get scene feature information
        context = self.context(x)

        # Including Context:
        context_h = self.fc_h(context)
        context_h = context_h.contiguous().view(N, 1, self.LSTMs_isz[0])
        local_feats_h = local_feats.contiguous().view(N, W_lf, self.LSTMs_isz[0])
        lstm_input_h = torch.cat((context_h, local_feats_h), dim=1)
         
        # Horizontal BLSTM_1
        output_h, _ = self.blstm_h_1(lstm_input_h)
        # Remove the context from the output (this is included in the other values through cell memory)
        output_h = output_h[:,1:,:]
        # Resize the output to (C, H, W)
        output_h = output_h.contiguous().view(N, 2*128, H_lf, W_lf)

        # Including Context:
        context_v = self.fc_v(context)
        context_v = context_v.contiguous().view(N, 1, self.LSTMs_isz[1])
        output_h  = output_h.contiguous().view(N, H_lf, self.LSTMs_isz[1])
        lstm_input_hv = torch.cat((context_v, output_h), dim=1)
        
        # Vertical BLSTM_1
        output_hv, _ = self.blstm_v_1(lstm_input_hv)
        # Remove the context from the output (this is included in the other values through cell memory)
        output_hv = output_hv[:,1:,:]
        # Resize the output to (C, H, W)
        output_hv = output_hv.contiguous().view(N, 2*128, H_lf, W_lf)

        # Horizontal BLSTM_2
        lstm_input_hvh = torch.cat((context_h, output_hv), dim=1)
        output_hvh, _ = self.blstm_h_2(lstm_input_hvh)
        # Remove the context from the output (this is included in the other values through cell memory)
        output_hvh = output_hvh[:,1:,:]
        # Resize the output to (C, H, W)
        output_hvh = output_hvh.contiguous().view(N, 2*128, H_lf, W_lf)

        # Vertical BLSTM_2
        lstm_input_hvhv = torch.cat((context_v, output_hvh), dim=1)
        output_hvhv, _ = self.blstm_v_2(lstm_input_hvhv)
        # Remove the context from the output (this is included in the other values through cell memory)
        output_hvhv = output_hvhv[:,1:,:]
        # Resize the output to (C, H, W)
        output_hvhv = output_hvhv.contiguous().view(N, 2*128, H_lf, W_lf)

        # Reduce channel dimension to 1
        output_conv = self.last_conv(output_hvhv)
        
        N, C, _, _, = output_conv.size()
        
        # Upsampling
        output_upsampled = nn.functional.interpolate(output_conv, size=self.input_dim, mode='bilinear', align_corners=False) # align_corners=False assumed, default behaviour was changed from True to False from pytorch 0.3.1 to 0.4
        
        # Softmax scoring
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
