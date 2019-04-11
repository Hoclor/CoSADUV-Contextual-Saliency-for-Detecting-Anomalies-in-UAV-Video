"""Data utility functions."""
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
from scipy.misc import imresize
from scipy.ndimage import imread
from skimage import io, transform

import cv2

try:
    import nvvl
    nvvl_is_available = True
except ImportError:
    nvvl_is_available = False


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
    """Salicon dataset, loaded from image files and dynamically resized as specified"""
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


#TODO implement this class
class _UAV123Data(data.Dataset):
    """NotImplemented: UAV123 dataset, loaded from image files and dynamically resized as specified"""
    def __init__(self, root_dir, mean_image_name, segments, img_size=(480, 640)):
        return NotImplementedError
        self.root_dir = root_dir
        self.segments = segments
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


def prepare_nvvl_UAV123_Dataset(root_dir, section, shuffle=False, sequence_length=150, img_size=(480, 640)):
    if not nvvl_is_available:
        return ModuleNotFoundError("nvvl is not available.")
    # Get a list of videos for this section
    if os.name == 'posix':
        # Unix
        videos = os.listdir(os.path.join(root_dir, 'UAV123', section))
        videos.remove('targets') # Remove the targets folder
    else:
        # Windows (os.name == 'nt')
        with os.scandir(os.path.join(root_dir, 'UAV123', section)) as folder_iterator:
            videos = [folder_object.name for folder_object in list(folder_iterator)]
            videos.remove('targets') # Remove the targets folder

    if shuffle:
        # Shuffle the order of videos (so input_videos and target_videos are ordered in the same way)
        random.shuffle(videos)
    
    # List of input videos
    input_videos = [os.path.join(root_dir, 'UAV123', section, name) for name in videos]
    # List of target videos
    target_videos = [os.path.join(root_dir, 'UAV123', section, 'targets', name) for name in videos]

    # Define the processing to be applied to the data
    processing = {
        "input": nvvl.ProcessDesc(
            width=img_size[1], # Scale image to the given width
            height=img_size[0], # Scale image to the given height
            normalized=True # Normalize image values from [0, 255] to [0, 1]
        )
    }
    # Create the NVVL Dataset for the input data
    input_data = nvvl.VideoDataset(filenames=input_videos, sequence_length=sequence_length, processing=processing)
    # Create the NVVL Dataset for the targets
    targets = nvvl.VideoDataset(filenames=target_videos, sequence_length=sequence_length, processing=processing)

    return input_data, targets


##### External retrieval functions #####


def get_SALICON_datasets(root_dir, mean_image_name, img_size=(480, 640)):
    """Returns a SALICON dataset, split into training, validation, and test sets."""
    train_data = SaliconData(root_dir, mean_image_name, 'train', img_size)
    val_data = SaliconData(root_dir, mean_image_name, 'val', img_size)
    test_data = SaliconData(root_dir, mean_image_name, 'test', img_size)
    
    return (train_data, val_data, test_data)

def _get_UAV123_datasets(root_dir, mean_image_name, splits=[0.6, 0.2, 0.2], sequence_lengths = 300, img_size=(480, 640)):
    """NotImplemented: Returns a UAV123 dataset, split into training, validation, and test sets
    as specified by 'segments' dictionary in this function.
    """
    return NotImplementedError
    segments = {
        'train': [],
        'val': [],
        'test': []
    }

    # Proportions of train/val/test split must sum to 1
    assert(sum(splits) == 1)

    # Randomly create the train/val/test split

    # List of all segments
    if os.name == 'posix':
        # Unix
        segment_list = os.listdir(os.path.join(root_dir, 'ground_truth', 'UAV123'))
    else:
        # Windows (os.name == 'nt')
        with os.scandir(os.path.join(root_dir, 'ground_truth', 'UAV123')) as folder_iterator:
            segment_list = [folder_object.name for folder_object in list(folder_iterator)]
    
    # Randomly split this list into train/val/test sets
    random.shuffle(segment_list)
    index_one = int(splits[0] * len(segment_list))
    index_two = int(index_one + splits[1] * len(segment_list))

    # Assign the segments
    segments['train'] = segment_list[:index_one]
    segments['val'] = segment_list[index_one:index_two]
    segments['test'] = segment_list[index_two:]

    train_data = _UAV123Data(root_dir, mean_image_name, segments['train'], img_size)
    val_data   = _UAV123Data(root_dir, mean_image_name, segments['val'], img_size)
    test_data  = _UAV123Data(root_dir, mean_image_name, segments['test'], img_size)
    
    return (train_data, val_data, test_data)

def get_nvvl_UAV123_datasets(root_dir, mean_image_name, shuffle=False, sequence_length = 150, img_size=(480, 640)):
    """Returns a UAV123 dataset using the NVVL dataset class."""
    if not nvvl_is_available:
        return ModuleNotFoundError("nvvl is not available.")
    train_data, train_targets = prepare_nvvl_UAV123_Dataset(root_dir, 'train', shuffle=shuffle, sequence_length=sequence_length, img_size=img_size)
    val_data, val_targets   = prepare_nvvl_UAV123_Dataset(root_dir, 'val', shuffle=shuffle, sequence_length=sequence_length, img_size=img_size)
    test_data, test_targets  = prepare_nvvl_UAV123_Dataset(root_dir, 'test', shuffle=shuffle, sequence_length=sequence_length, img_size=img_size)

    mean_image = np.load(os.path.join(root_dir, mean_image_name))
    mean_image = cv2.resize(mean_image, (img_size[1], img_size[0])) # Resize the mean_image to the correct size
    mean_image = mean_image.astype(np.float32)/255. # Convert to [0, 1] (float)
    mean_image = torch.from_numpy(mean_image).transpose(0, 1).transpose(1, 2) # Convert to tensor and from (H, W, C) to (C, H, W)
    
    return (train_data, train_targets), (val_data, val_targets), (test_data, test_targets), mean_image


##### Dataloader preparation functions #####


def get_dataloader(dataset_type, dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True):
    if dataset_type.upper() == 'SALICON':
        return get_torch_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory)
    elif dataset_type.upper() == 'UAV123':
        if type(dataset) != tuple and type(dataset) != list:
            return TypeError("dataset variable must be a tuple or list of two nvvl.VideoDataset if dataset type is UAV123")
        # Ignore shuffle, num_workers, pin_memory settings as they are not applicable (shuffle is handled in the get_dataset function)
        return get_nvvl_dataloader(dataset[0], dataset[1], batch_size)
    else:
        return NotImplementedError

def get_nvvl_dataloader(input_dataset, target_dataset, batch_size):
    if not nvvl_is_available:
        return ModuleNotFoundError("nvvl is not available.")
    # Prepare the nvvl loaders
    input_loader = nvvl.VideoLoader(input_dataset, batch_size=batch_size, shuffle=False)
    target_loader = nvvl.VideoLoader(target_dataset, batch_size=batch_size, shuffle=False)

    # Zip the loaders so they produce output as (input, target)
    data_loader = zip(input_loader, target_loader)
    return data_loader

def get_torch_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
