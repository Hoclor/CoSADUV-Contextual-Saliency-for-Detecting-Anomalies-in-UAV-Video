"""Data utility functions."""
import os
import sys

import numpy as np
import torch
import torch.utils.data as data

from scipy.ndimage import imread
from scipy.misc import imresize

from skimage import io, transform

import pickle

import cv2

##### Transformation classes #####

class Rescale(object):
    """Rescale the image and target in a sample to a given size.
    Source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, target = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        tar = transform.resize(target, (new_h, new_w))

        return img, tar
    
class Normalize(object):
    """Normalize the image in a sample by subtracting a mean image."""

    def __init__(self, mean_image):
        self.mean_image = mean_image

    def __call__(self, sample):
        image, target = sample

        if image.shape[:2] != self.mean_image.shape[:2]:
            self.mean_image = transform.resize(self.mean_image, image.shape[:2])
        
        image = image - self.mean_image

        return image, target

class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    Source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __call__(self, sample):
        image, target = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        tar = target.transpose((2, 0, 1))
        return image, tar

##### Dataset classes #####

class SaliconData(data.Dataset):
    """ Salicon dataset, loaded from image files and dynamically resized as specified"""
    def __init__(self, root_dir, mean_image_name, section, img_size=(480, 640)):
        self.root_dir = root_dir
        self.section = section.lower()
        self.img_size = img_size # Height, Width
        self.mean_image = np.load(os.path.join(self.root_dir, mean_image_name))
        self.mean_image = cv2.resize(self.mean_image, (self.img_size[1], self.img_size[0])) # Resize the mean_image to the correct size
        self.mean_image = self.mean_image.astype(np.float32)/255. # Convert to [0, 1] (float)
        
        # Create the list of images in this section in root_dir
        if os.name == 'posix':
            # Unix
            file_names = os.listdir(os.path.join(self.root_dir, 'images'))
            # Filter the file_names so we only get the files in this section
            file_names = [name for name in file_names if section in name]
        else:
            # Windows (os.name == 'nt')
            with os.scandir(os.path.join(self.root_dir, 'images')) as file_iterator:
                file_names = [file_object.name for file_object in list(file_iterator) if section in file_object.name]
        self.image_list = sorted(file_names)
    
    def __getitem__(self, index):
        # Load the image of given index, and its fixation map (if section == Test, return fully black image as fixation map as they do not exist for test images)
        img_name = os.path.join(self.root_dir, 'images', self.image_list[index])
        
        image = imread(img_name)
        image = imresize(image, self.img_size)
        # If image is Grayscale convert it to RGB
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, 2)
        
        # Normalize image by subtracting mean_image, convert from [0, 255] (int) to [0, 1] (float), and from [H, W, C] to [C, H, W]
        image = (image.astype(np.float32)/255. - self.mean_image).transpose(2, 0, 1)
        
        if self.section == 'test':
            fix_map_name = 'None'
            fix_map = np.zeros(self.img_size, dtype=np.float32)
        else:
            fix_map_name = os.path.join(self.root_dir, 'fixation maps', self.section, self.image_list[index][:-4]) + '.png' # Images are .jpg, fixation maps are .png
            
            fix_map = imread(fix_map_name)
            fix_map = imresize(fix_map, self.img_size)
            
            # Normalize fixation map by converting from [0, 255] (int) to [0, 1] (float), and from [H, W, C] to [C, H, W]
            fix_map = (fix_map.astype(np.float32)/255.)
        
        # Convert image and fix_map to torch tensors
        image = torch.from_numpy(image)
        fix_map = torch.from_numpy(fix_map)
        
        # Return the image and the fix_map
        return image, fix_map
        
    def __len__(self):
        return len(self.image_list)


##### External functions #####

##### External retrieval functions #####

def get_SALICON_datasets(root_dir, mean_image_name, img_size=(480, 640)):
    """ Returns a SALICON dataset, split into training, validation, and test sets."""
    train_data = SaliconData(root_dir, mean_image_name, 'train', img_size)
    val_data = SaliconData(root_dir, mean_image_name, 'val', img_size)
    test_data = SaliconData(root_dir, mean_image_name, 'test', img_size)
    
    return (train_data, val_data, test_data)
