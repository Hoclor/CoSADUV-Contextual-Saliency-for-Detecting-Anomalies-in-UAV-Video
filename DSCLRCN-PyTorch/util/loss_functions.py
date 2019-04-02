import torch
import numpy as np

# Normalized Scanpath Saliency - x, y should be individual images (not batches)
def NSS(x, y):
    """
        Computes the Normalized Scanpath Saliency between x (output of a model)
        and y (label). x and y are assumed to be individual images as torch tensors.
    """
    # Normalize x
    x = (x - x.mean())/x.std()
    # Compute the element-wise multiplication of x and y
    scanpath = x * y
    # Compute the sum of the scanpath divided by the sum of values in y as NSS
    nss = scanpath.sum()/y.sum() if y.sum() > 0 else scanpath.sum()
    if y.sum() <= 0:
        print("Error: unexpected sum of ground truth: {}".format(y.sum()))
    return nss

def NSS_loss(x, y):
        """
        Computes the Normalized Scanpath Saliency loss between x (output of a model) and y (label).
        x and y are assumed to be torch tensors, either individual images or batches.
        """
        # Check if dimensionality of x is 3. If so, apply NSS loss on each individual set in the batch, and return avg loss
        if len(x.shape) == 2:
            return -NSS(x, y)
        else:
            batch_sum = 0
            for i in range(x.shape[0]):
                batch_sum += -NSS(x[i, :, :], y[i, :, :])
            return batch_sum/x.shape[0]
        
def NSS_2(x, y):
    # normalize saliency map
    sal_map = (x - torch.mean(x))/torch.std(x)
    # mean value at fixation locations
    sal_map = sal_map.masked_select(y > 0)
    score = torch.mean(sal_map)
    return score

def NSS_loss_2(x, y):
        """
        Computes the Normalized Scanpath Saliency loss between x (output of a model) and y (label).
        x and y are assumed to be torch tensors, either individual images or batches.
        """
        # Check if dimensionality of x is 3. If so, apply NSS loss on each individual set in the batch, and return avg loss
        if len(x.shape) == 2:
            return -NSS_2(x, y)
        else:
            batch_sum = 0
            for i in range(x.shape[0]):
                batch_sum += -NSS_2(x[i, :, :], y[i, :, :])
            return batch_sum/x.shape[0]

# Pearson Cross Correlation
def PCCLoss_torch(x, y):
    """Computes Pearson Cross Correlation loss
    :param x: prediction
    :param y: label
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    
    cc = torch.sum(vx*y) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
    # since cc is in [-1, 1], and 0 is 'bad' and close to -1 or 1 is 'good', return the abs value of cc
    cc = abs(cc)
    # actually return 1 -  cc, as we need to return a loss (since 1 is good, we return loss as 1 - cc)
    loss = 1 - cc
    
    return loss


# Define the Pearson Cross Correlation loss function, using numpy:
def PCC_loss_numpy(x, y):
    """Computes Pearson Cross Correlation loss
    :param x: prediction
    :param y: label
    """
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    
    loss = np.sum(vx*y) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    
    return loss
