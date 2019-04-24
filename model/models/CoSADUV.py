"""Model for Contextual Saliency for Anomaly Detection in UAV Video (CoSADUV)
Based on the Deep Spatial Contextual Long-term Recurrent Convolutional Network model
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision

from models.cnn_vgg16.local_cnn import LocalFeatsCNN
from models.places_vgg16.places_cnn import PlacesCNN
from models.segmentation_resnet50.segmentation_nn import SegmentationNN


class CoSADUV(nn.Module):
    def __init__(self, input_dim=(480, 640), local_feats_net="Seg"):
        super(CoSADUV, self).__init__()
        self.temporal = True

        self.input_dim = input_dim

        # Hidden size of the LSTMs
        # LSTM_1 hidden size: 128
        # LSTM_2 hidden size: 128
        # LSTM_3 hidden size: 128
        # LSTM_4 hidden size: 128
        self.pixel_LSTMs_hsz = (128, 128, 128, 128)

        # Input size of the LSTMs
        # LSTM_1 input size: channel of local_feats output (512)
        # LSTM_2 input size: 256 (2 * 128 as LSTMs output 128 values, *2 for bLSTM)
        # LSTM_3 input size: 256 (same reason as above)
        # LSTM_4 input size: 256 (same reason as above)
        self.pixel_LSTMs_isz = (
            512,
            2 * self.pixel_LSTMs_hsz[0],
            2 * self.pixel_LSTMs_hsz[1],
            2 * self.pixel_LSTMs_hsz[2],
        )

        if local_feats_net == "Seg":
            self.local_feats = SegmentationNN()
        else:
            self.local_feats = LocalFeatsCNN()

        self.scene_context = PlacesCNN(input_dim=input_dim)
        self.scene_context_fc_1 = nn.Linear(128, self.pixel_LSTMs_isz[0])
        self.scene_context_fc_rest = nn.Linear(128, self.pixel_LSTMs_isz[1])

        # Constructing LSTMs:
        self.pixel_blstm_h_1 = nn.LSTM(
            input_size=self.pixel_LSTMs_isz[0],
            hidden_size=self.pixel_LSTMs_hsz[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pixel_blstm_v_1 = nn.LSTM(
            input_size=self.pixel_LSTMs_isz[1],
            hidden_size=self.pixel_LSTMs_hsz[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pixel_blstm_h_2 = nn.LSTM(
            input_size=self.pixel_LSTMs_isz[2],
            hidden_size=self.pixel_LSTMs_hsz[2],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pixel_blstm_v_2 = nn.LSTM(
            input_size=self.pixel_LSTMs_isz[3],
            hidden_size=self.pixel_LSTMs_hsz[3],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Initialize the biases of the forget gates to 1 for all blstms
        for blstm in [
            self.pixel_blstm_h_1,
            self.pixel_blstm_v_1,
            self.pixel_blstm_h_2,
            self.pixel_blstm_v_2,
        ]:
            # Below code taken from:
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
            for names in blstm._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(blstm, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

        # Last conv to move to one channel
        self.last_conv = nn.Conv2d(2 * self.pixel_LSTMs_hsz[-1], 1, 1)

        # LSTM applied to the sequence in time domain, one hidden cell per pixel
        self.temporal_LSTM = nn.LSTM(
            input_size=input_dim[0] // 8 * input_dim[1] // 8,
            hidden_size=input_dim[0] // 8 * input_dim[1] // 8,
            num_layers=1,
            batch_first=True,
        )

        # The hidden state of the temporal LSTM
        self.temporal_LSTM_state = None
        self.stored_temporal_state = False

        # # softmax
        self.score = nn.Sigmoid()

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
        raw_scene_context = self.scene_context(x)
        # Create context input into BLSTM_1
        scene_context_first = self.scene_context_fc_1(raw_scene_context)
        # Create context input into BLSTM_[2,3,4]
        scene_context_rest = self.scene_context_fc_rest(raw_scene_context)

        # Horizontal BLSTM_1
        # local_feats_h shape: (N, H, W, C)
        local_feats_h = local_feats.transpose(1, 2).transpose(2, 3).contiguous()
        # Context shape: (N, 1, C)
        scene_context_h = scene_context_first.contiguous().view(
            N, 1, self.pixel_LSTMs_isz[0]
        )
        # Loop over local_feats one row at a time:
        # split(1,1) splits it into individual rows,
        # squeeze removes the row dimension
        rows = []
        for row in local_feats_h.split(1, 1):  # row shape (N, 1, W, C)
            # Add context to the start and end of the row
            row = row.squeeze(1)
            row = torch.cat((scene_context_h, row, scene_context_h), dim=1)
            # FIXME: Error here if using PyTorch version >=1.0:
            # BLSTM returns nan for all values in row but context_h (first and last)
            result, _ = self.pixel_blstm_h_1(row)
            result = result[:, 1:-1, :]
            rows.append(result)
        # Reconstruct the image by stacking the rows
        output_h = torch.stack(rows, dim=1)  # Shape (N, H, W, C)
        del rows, row, result

        # Vertical BLSTM_1
        # Context shape: (N, 1, C)
        scene_context_v = scene_context_rest.contiguous().view(
            N, 1, self.pixel_LSTMs_isz[1]
        )
        # Loop over local_feats one column at a time:
        # split(1,2) splits it into individual columns,
        # squeeze removes the column dimension
        cols = []
        for col in output_h.split(1, 2):  # col shape (N, H, 1, C)
            # Add context to the start and end of the col
            col = col.squeeze(2)
            col = torch.cat((scene_context_v, col, scene_context_v), dim=1)
            result, _ = self.pixel_blstm_v_1(col)
            result = result[:, 1:-1, :]
            cols.append(result)
        # Reconstruct the image by stacking the columns
        output_hv = torch.stack(cols, dim=2)  # Shape (N, H, W, C)
        del cols, col, result

        # Horizontal BLSTM_2
        # Context shape: (N, 1, C)
        scene_context_h_2 = scene_context_rest.contiguous().view(
            N, 1, self.pixel_LSTMs_isz[2]
        )
        # Loop over local_feats one row at a time:
        # split(1,1) splits it into individual rows,
        # squeeze removes the row dimension
        rows = []
        for row in output_hv.split(1, 1):  # row shape (N, 1, W, C)
            # Add context to the start and end of the row
            row = row.squeeze(1)
            row = torch.cat((scene_context_h_2, row, scene_context_h_2), dim=1)
            result, _ = self.pixel_blstm_h_2(row)
            result = result[:, 1:-1, :]
            rows.append(result)
        # Reconstruct the image by stacking the rows
        output_hvh = torch.stack(rows, dim=1)  # Shape (N, H, W, C)
        del rows, row, result

        # Vertical BLSTM_2
        # Context shape: (N, 1, C)
        scene_context_v = scene_context_rest.contiguous().view(
            N, 1, self.pixel_LSTMs_isz[3]
        )
        # Loop over local_feats one column at a time:
        # split(1,2) splits it into individual columns,
        # squeeze removes the column dimension
        cols = []
        for col in output_hvh.split(1, 2):  # col shape (N, H, 1, C)
            # Add context to the start and end of the col
            col = col.squeeze(2)
            col = torch.cat((scene_context_v, col, scene_context_v), dim=1)
            result, _ = self.pixel_blstm_v_2(col)
            result = result[:, 1:-1, :]
            cols.append(result)
        # Reconstruct the image by stacking the columns
        output_hvhv = torch.stack(cols, dim=2)  # Shape (N, H, W, C)
        output_hvhv = output_hvhv.transpose(1, 3).transpose(2, 3)  # Shape (N, C, H, W)
        del cols, col, result

        # Reduce channel dimension to 1
        output_conv = self.last_conv(output_hvhv)  # Shape (N, 1, H, W)

        N, _, H, W, = output_conv.size()

        output_conv = output_conv.contiguous().view(N, 1, H * W)

        # Apply the temporal LSTM
        # Give the current temporal state as input if there is one stored
        if self.stored_temporal_state:
            output_temporal, self.temporal_LSTM_state = self.temporal_LSTM(
                output_conv, self.temporal_LSTM_state
            )
        else:
            output_temporal, self.temporal_LSTM_state = self.temporal_LSTM(output_conv)
            self.stored_temporal_state = True

        output_temporal = output_temporal.contiguous().view(N, 1, H, W)

        # Upsampling - nn.functional.interpolate does not exist in < 0.4.1,
        # but upsample is deprecated in > 0.4.0
        if torch.__version__ == "0.4.0":
            output_upsampled = nn.functional.upsample(
                output_temporal,
                size=self.input_dim,
                mode="bilinear",
                align_corners=True,
            )
        else:
            # align_corners=False assumed, default behaviour was changed
            # from True to False from pytorch 0.3.1 to 0.4
            output_upsampled = nn.functional.interpolate(
                output_temporal,
                size=self.input_dim,
                mode="bilinear",
                align_corners=True,
            )

        # Sigmoid scoring - project each pixel's value into probability space (0, 1)
        output_score = self.score(output_upsampled)

        return output_score

    def clear_temporal_state(self):
        self.temporal_LSTM_state = None
        self.stored_temporal_state = False

    def detach_temporal_state(self):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(self.temporal_LSTM_state) == type(None):
            pass
        else:
            self.temporal_LSTM_state[0].detach_()
            self.temporal_LSTM_state[1].detach_()
            self.temporal_LSTM_state = (
                self.temporal_LSTM_state[0].detach(),
                self.temporal_LSTM_state[1].detach(),
            )

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
