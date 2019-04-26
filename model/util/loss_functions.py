import numpy as np
import torch


# Normalized Scanpath Saliency
def NSS_loss(x, y):
    """
        Computes the Normalized Scanpath Saliency loss between x (output of a model)
        and y (label).
        x and y are assumed to be torch tensors, either individual images or batches.
        """
    # If dimensionality of x is 2, insert a singleton batch dimension
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    # Loop over each image in the batch, apply NSS, return the average
    loss = 0
    for i in range(x.shape[0]):
        x_i, y_i = x[i, :, :], y[i, :, :]
        # Normalize x_i
        x_i = (x_i - x_i.mean()) / x_i.std()
        # Compute the element-wise multiplication of x_i and y_i
        scanpath = x_i * y_i
        # Compute the sum of the scanpath divided by the sum of values in y_i as NSS
        loss += scanpath.sum() / y_i.sum() if y_i.sum() > 0 else scanpath.sum()
        if y_i.sum() == 0:
            print("Error: zero sum of ground truth")
    # Return the -ve avg NSS score
    return -1 * loss / x.shape[0]


def NSS_loss_2(x, y):
    """
        Computes the Normalized Scanpath Saliency loss between x (output of a model)
        and y (label).
        x and y are assumed to be torch tensors, either individual images or batches.
        """
    # If dimensionality of x is 2, insert a singleton batch dimension
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    # Loop over each image in the batch, apply NSS, return the average
    loss = 0
    for i in range(x.shape[0]):
        x_i, y_i = x[i, :, :], y[i, :, :]
        # normalize saliency map
        sal_map = (x_i - x_i.mean()) / x_i.std()
        # mean value at fixation locations
        sal_map = sal_map.masked_select(y_i > 0)
        loss += sal_map.mean()
    # Return the -ve avg NSS score
    return -1 * loss / x.shape[0]


# Cross entropy combined with MAE loss
def CE_MAE_loss(x, y):
    """ Computes loss, given prediction x and target y, as the sum of Cross Entropy loss
    and MAE loss.
    x and y must be 2 dimensional tensors corresponding to an image, of the same size.

    Used by: https://paperswithcode.com/paper/pyramid-dilated-deeper-convlstm-for-video
    to train model for video saliency evaluation
    """
    return torch.nn.functional.binary_cross_entropy(x, y) + torch.nn.functional.l1_loss(
        x, y
    )


def CE_loss(x, y):
    """Computes the Cross Entropy loss of the given prediction x and target y"""
    return torch.nn.functional.binary_cross_entropy(x, y)


def MAE_loss(x, y):
    """Computes the MAE loss of the given prediction x and target y"""
    return torch.nn.functional.l1_loss(x, y)


def KLDiv_loss(x, y):
    """Wrapper for PyTorch's KLDivLoss function"""
    x_log = x.log()
    return torch.nn.functional.kl_div(x_log, y)


# Pearson Cross Correlation
def PCC_loss(x, y):
    """Computes Pearson Cross Correlation loss
    :param x: prediction
    :param y: label
    """
    # If dimensionality of x is 2, insert a singleton batch dimension
    if len(x.shape) == 2:
        x = x.squeeze(0)
        y = y.squeeze(0)
    # Loop over each image in the batch, apply PCC, return the average
    loss = 0
    for i in range(x.shape[0]):
        x_i, y_i = x[i, :, :], y[i, :, :]
        # Subtract mean from x and y
        vx = x_i - torch.mean(x_i)
        vy = y_i - torch.mean(y_i)
        cc = torch.sum(vx * y_i) / (
            torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        )
        # Loss is equal to 1 - the absolute value of cc
        loss += 1 - abs(cc)
    return loss / x.shape[0]
