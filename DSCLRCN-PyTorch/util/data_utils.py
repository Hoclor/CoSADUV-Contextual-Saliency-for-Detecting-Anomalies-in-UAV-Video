"""Data utility functions."""
import os
import sys

import numpy as np
import torch
import torch.utils.data as data

from scipy.ndimage import imread
from scipy.misc import imresize

import pickle

class OverfitSampler(object):
    """
    Sample dataset to overfit.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class SaliconData(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        fix_map = self.y[index]

        img = torch.from_numpy(img)
        fix_map = torch.from_numpy(fix_map)
        return img, fix_map

    def __len__(self):
        return len(self.y)

    
def get_SALICON_datasets():
    """
    Load and preprocess the CIFAR-10 dataset.
    """

    mean_image = np.load('C:/Users/simon/Downloads/Project Datasets/SALICON/Transformed/mean_image.npy').astype(np.float32)/255.
    
    with open('C:/Users/simon/Downloads/Project Datasets/SALICON/Transformed/train_datadict.pickle', 'rb') as f:
        train_data = pickle.load(f)
    X_train = [(image.astype(np.float32)/255. - mean_image).transpose(2,0,1) for image in train_data['images']]
    y_train = [fix_map.astype(np.float32)/255. for fix_map in train_data['fix_maps']]
    # Restart the line
    sys.stdout.write('Progress: 50%\r')
    sys.stdout.flush()

    with open('C:/Users/simon/Downloads/Project Datasets/SALICON/Transformed/val_datadict.pickle', 'rb') as f:
        val_data = pickle.load(f)
    X_val = [(image.astype(np.float32)/255. - mean_image).transpose(2,0,1) for image in val_data['images']]
    y_val = [fix_map.astype(np.float32)/255. for fix_map in val_data['fix_maps']]
    # Restart the line
    sys.stdout.write('Progress: 75%\r')
    sys.stdout.flush()

    with open('C:/Users/simon/Downloads/Project Datasets/SALICON/Transformed/test_datadict.pickle', 'rb') as f:
        test_data = pickle.load(f)
    X_test = [(image.astype(np.float32)/255. - mean_image).transpose(2,0,1) for image in test_data['images']]
    y_test = [fix_map.astype(np.float32)/255. for fix_map in test_data['fix_maps']]
    # Restart the line
    sys.stdout.write('Progress: 100%\r')
    sys.stdout.flush()
    
    return (SaliconData(X_train, y_train),
            SaliconData(X_val, y_val),
            SaliconData(X_test, y_test))

def get_SALICON_subset(file_name):
    
    mean_image = np.load('Dataset/Transformed/mean_image.npy').astype(np.float32)/255.

    with open('Dataset/Transformed/'+file_name, 'rb') as f:
        data = pickle.load(f)
    X = [image.astype(np.float32)/255. - mean_image for image in data['images']]
    y = [fix_map.astype(np.float32)/255. for fix_map in data['fix_maps']]
            
    return SaliconData(X, y)
    
    