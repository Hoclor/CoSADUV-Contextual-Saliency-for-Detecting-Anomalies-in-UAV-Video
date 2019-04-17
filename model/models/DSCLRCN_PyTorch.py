"""Deep Spatial Contextual Long-term Recurrent Convolutional Network"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_vgg16.local_cnn import LocalFeatsCNN
from models.places_vgg16.places_cnn import PlacesCNN
from models.segmentation_resnet50.segmentation_nn import SegmentationNN

import numpy as np


class DSCLRCN(nn.Module):
    def __init__(self, input_dim=(480, 640), local_feats_net="CNN"):
        super(DSCLRCN, self).__init__()

        self.input_dim = input_dim

        # Hidden size of the LSTMs
        # LSTM_1 hidden size: 128
        # LSTM_2 hidden size: 128
        # LSTM_3 hidden size: 128
        # LSTM_4 hidden size: 128
        self.LSTMs_hsz = (128, 128, 128, 128)

        # Input size of the LSTMs
        # LSTM_1 input size: channel of local_feats output (512)
        # LSTM_2 input size: 256 (2 * 128 as LSTMs output 128 values, *2 for bLSTM)
        # LSTM_3 input size: 256 (same reason as above)
        # LSTM_4 input size: 256 (same reason as above)
        self.LSTMs_isz = (
            512,
            2 * self.LSTMs_hsz[0],
            2 * self.LSTMs_hsz[1],
            2 * self.LSTMs_hsz[2],
        )

        if local_feats_net == "Seg":
            self.local_feats = SegmentationNN()
        else:
            self.local_feats = LocalFeatsCNN()

        self.context = PlacesCNN(input_dim=input_dim)
        self.context_fc_1 = nn.Linear(128, self.LSTMs_isz[0])
        self.context_fc_rest = nn.Linear(128, self.LSTMs_isz[1])

        # Constructing LSTMs:
        self.blstm_h_1 = nn.LSTM(
            input_size=self.LSTMs_isz[0],
            hidden_size=self.LSTMs_hsz[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.blstm_v_1 = nn.LSTM(
            input_size=self.LSTMs_isz[1],
            hidden_size=self.LSTMs_hsz[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.blstm_h_2 = nn.LSTM(
            input_size=self.LSTMs_isz[2],
            hidden_size=self.LSTMs_hsz[2],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.blstm_v_2 = nn.LSTM(
            input_size=self.LSTMs_isz[3],
            hidden_size=self.LSTMs_hsz[3],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Initialize the biases of the forget gates to 1 for all blstms
        for blstm in [self.blstm_h_1, self.blstm_v_1, self.blstm_h_2, self.blstm_v_2]:
            # Below code taken from:
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
            for names in blstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(blstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

        # Last conv to move to one channel
        self.last_conv = nn.Conv2d(2 * self.LSTMs_hsz[3], 1, 1)

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
        H, W = self.input_dim

        # Get local feature map
        local_feats = self.local_feats(x)  # Shape (N, C, H, W)
        H_lf, W_lf = local_feats.size()[2:]

        # Get scene feature information
        context = self.context(x)
        # Create context input into BLSTM_1
        context_1 = self.context_fc_1(context)
        # Create context input into BLSTM_[2,3,4]
        context_rest = self.context_fc_rest(context)

        # Horizontal BLSTM_1
        # local_feats_h shape: (N, H, W, C)
        local_feats_h = local_feats.transpose(1, 2).transpose(2, 3).contiguous()
        # Context shape: (N, 1, C)
        context_h = context_1.contiguous().view(N, 1, self.LSTMs_isz[0])
        # Loop over local_feats one row at a time:
        # split(1,1) splits it into individual rows,
        # squeeze removes the row dimension
        rows = []
        for row in local_feats_h.split(1, 1):  # row shape (N, 1, W, C)
            # Add context to the start and end of the row
            row = row.squeeze(1)
            row = torch.cat((context_h, row, context_h), dim=1)
            # FIXME: Error here if using PyTorch version >=1.0:
            # BLSTM returns nan for all values in row but context_h (first and last)
            result, _ = self.blstm_h_1(row)
            result = result[:, 1:-1, :]
            rows.append(result)
        # Reconstruct the image by stacking the rows
        output_h = torch.stack(rows, dim=1)  # Shape (N, H, W, C)
        del rows, row, result

        # Vertical BLSTM_1
        # Context shape: (N, 1, C)
        context_v = context_rest.contiguous().view(N, 1, self.LSTMs_isz[1])
        # Loop over local_feats one column at a time:
        # split(1,2) splits it into individual columns,
        # squeeze removes the column dimension
        cols = []
        for col in output_h.split(1, 2):  # col shape (N, H, 1, C)
            # Add context to the start and end of the col
            col = col.squeeze(2)
            col = torch.cat((context_v, col, context_v), dim=1)
            result, _ = self.blstm_v_1(col)
            result = result[:, 1:-1, :]
            cols.append(result)
        # Reconstruct the image by stacking the columns
        output_hv = torch.stack(cols, dim=2)  # Shape (N, H, W, C)
        del cols, col, result

        # Horizontal BLSTM_2
        # Context shape: (N, 1, C)
        context_h_2 = context_rest.contiguous().view(N, 1, self.LSTMs_isz[2])
        # Loop over local_feats one row at a time:
        # split(1,1) splits it into individual rows,
        # squeeze removes the row dimension
        rows = []
        for row in output_hv.split(1, 1):  # row shape (N, 1, W, C)
            # Add context to the start and end of the row
            row = row.squeeze(1)
            row = torch.cat((context_h_2, row, context_h_2), dim=1)
            result, _ = self.blstm_h_2(row)
            result = result[:, 1:-1, :]
            rows.append(result)
        # Reconstruct the image by stacking the rows
        output_hvh = torch.stack(rows, dim=1)  # Shape (N, H, W, C)
        del rows, row, result

        # Vertical BLSTM_2
        # Context shape: (N, 1, C)
        context_v = context_rest.contiguous().view(N, 1, self.LSTMs_isz[3])
        # Loop over local_feats one column at a time:
        # split(1,2) splits it into individual columns,
        # squeeze removes the column dimension
        cols = []
        for col in output_hvh.split(1, 2):  # col shape (N, H, 1, C)
            # Add context to the start and end of the col
            col = col.squeeze(2)
            col = torch.cat((context_v, col, context_v), dim=1)
            result, _ = self.blstm_v_2(col)
            result = result[:, 1:-1, :]
            cols.append(result)
        # Reconstruct the image by stacking the columns
        output_hvhv = torch.stack(cols, dim=2)  # Shape (N, H, W, C)
        output_hvhv = output_hvhv.transpose(1, 3).transpose(2, 3)  # Shape (N, C, H, W)
        del cols, col, result

        # Reduce channel dimension to 1
        output_conv = self.last_conv(output_hvhv)

        N, C, _, _, = output_conv.size()

        # Upsampling - nn.functional.interpolate does not exist in < 0.4.1,
        # but upsample is deprecated in > 0.4.0, so use this switch
        if torch.__version__ == "0.4.0":
            output_upsampled = nn.functional.upsample(
                output_conv, size=self.input_dim, mode="bilinear", align_corners=True
            )
        else:
            # align_corners=False assumed, default behaviour was changed
            # from True to False from pytorch 0.3.1 to 0.4
            output_upsampled = nn.functional.interpolate(
                output_conv, size=self.input_dim, mode="bilinear", align_corners=True
            )

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
        print("Saving model... %s" % path)
        torch.save(self, path)
