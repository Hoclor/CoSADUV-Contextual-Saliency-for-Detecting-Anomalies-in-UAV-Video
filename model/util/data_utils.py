"""Data utility functions."""
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as data

from scipy.ndimage import imread
from scipy.misc import imresize

from skimage import io, transform

import pickle

import cv2

class SaliconData(data.Dataset):
    """ Salicon dataset, loaded from image files and dynamically resized as specified"""
    def __init__(self, root_dir, mean_image_name, section, img_size=(96, 128)):
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

class VideoData(data.Dataset):
    """ A dataset of a single video, loaded from frame files and dynamically resized as specified.
    Returned by the VideoDataset class, but can be used independently.
    File structure should match description in /Dataset/UAV123/README.md
    """
    def __init__(self, root_dir, mean_image_name, section, video_name, duration=-1, img_size=(480, 640)):
        self.video_folder = os.path.join(root_dir, section, video_name)
        self.section = section.lower()
        self.img_size = img_size # Height, Width
        self.mean_image = np.load(os.path.join(root_dir, mean_image_name))
        self.mean_image = cv2.resize(self.mean_image, (self.img_size[1], self.img_size[0])) # Resize the mean_image to the correct size
        self.mean_image = self.mean_image.astype(np.float32)/255. # Convert to [0, 1] (float)
        
        # Create the list of frames for this video (in video_folder/frames)
        if os.name == 'posix':
            # Unix
            frame_list = os.listdir(os.path.join(self.video_folder, 'frames'))
            # Filter the list of frames so we only get image files (assumped to be .jpg or .png)
            frame_list = [name for name in frame_list if name.endswith('.jpg') or name.endswith('.png')]
        else:
            # Windows (os.name == 'nt')
            with os.scandir(os.path.join(self.video_folder, 'frames')) as frame_list:
                frame_list = [frame.name for frame in list(frame_list) if frame.name.endswith('.jpg') or frame.name.endswith('.png')]
        self.frame_list = sorted(frame_list)

        if duration > -1:
            # Slice the frame list at a random (valid) index
            start_index = random.randrange(0, len(self.frame_list) - duration)
            self.frame_list = self.frame_list[start_index:start_index+duration]

    
    def __getitem__(self, index):
        # Load the image of given index, and its target
        img_name = os.path.join(self.video_folder, 'frames', self.frame_list[index])
        
        image = imread(img_name)
        image = imresize(image, self.img_size)
        # If image is Grayscale convert it to RGB
        if len(image.shape) == 2:
            image = np.repeat(np.expand_dims(image, 2), 3, 2)
        
        # Normalize image by converting from [0,255] to [0,1], subtracting mean_image,
        # and reordering from [H, W, C] to [C, H, W]
        image = (image.astype(np.float32)/255.0 - self.mean_image).transpose(2, 0, 1)
        
        # Load the target
        target = imread(os.path.join(self.video_folder, 'targets', self.frame_list[index]))
        target = imresize(target, self.img_size)
        
        # Normalize target by converting from [0, 255] (int) to [0, 1] (float),
        # and reordering from [H, W, C] to [C, H, W]
        target = target.astype(np.float32)/255.
        
        # Convert image and target to torch tensors
        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        
        # Return the image and the target
        return image, target
        
    def __len__(self):
        return len(self.frame_list)


class VideoDataset(data.Dataset):
    """ Video dataset, loaded from frame files and dynamically resized as specified.

    Loads each folder in 'section' as a separate video: e.g., if section is 'train',
    then it loads 'train/video1', 'train/video2', 'train/video3' as 3 separate videos.

    This is done by creating a standard torch.utils.data.Dataset for each folder
    and supplying this when __getitem__ is called. Thus, the VideoData class can be
    iterated over to yield a list of video datasets, which in turn can be iterated over
    to yield each frame of the video (with its corresponding ground truth).
    """
    def __init__(self, root_dir, mean_image_name, section, duration=-1, img_size=(480, 640), loader_settings={}):
        self.root_dir = root_dir
        self.section = section.lower()
        self.img_size = img_size # Height, Width
        self.mean_image_name = mean_image_name
        self.duration = duration # Length of each sequence
        
        # Create the list of videos in this section in root_dir
        if os.name == 'posix':
            # Unix
            video_names = os.listdir(os.path.join(self.root_dir, section))
        else:
            # Windows (os.name == 'nt')
            with os.scandir(os.path.join(self.root_dir, section)) as file_iterator:
                video_names = [folder.name for folder in list(file_iterator)]
        video_names = sorted(video_names)

        batch_size = loader_settings['batch_size']
        num_workers = loader_settings['num_workers']
        pin_memory = loader_settings['pin_memory']

        # Produce a list of dataloaders of datasets, one for each video in video_names
        self.video_list = [
            data.DataLoader(
                VideoData(
                    self.root_dir, self.mean_image_name, self.section, video_name, self.duration, self.img_size
                ), 
                batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
            ) for video_name in video_names
        ]
    
    def __getitem__(self, index):
        # Return the dataset of the video of the given index
        return self.video_list[index]
        
    def __len__(self):
        # Return a count of how many videos there are
        return len(self.video_list)


def get_SALICON_datasets(root_dir, mean_image_name, img_size=(480, 640)):
    train_data = SaliconData(root_dir, mean_image_name, 'train', img_size)
    val_data = SaliconData(root_dir, mean_image_name, 'val', img_size)
    test_data = SaliconData(root_dir, mean_image_name, 'test', img_size)
    
    mean_image = np.load(os.path.join(root_dir, mean_image_name))
    mean_image = cv2.resize(mean_image, (img_size[1], img_size[0])) # Resize the mean_image to the correct size
    mean_image = mean_image.astype(np.float32)/255. # Convert to [0, 1] (float)
    
    return (train_data, val_data, test_data, mean_image)

def get_video_datasets(root_dir, mean_image_name, duration=-1, img_size=(480, 640), loader_settings={'batch_size': minibatchsize, 'num_workers': 8, 'pin_memory': True}):
    train_data = VideoDataset(root_dir, mean_image_name, 'train', duration=duration, img_size=img_size, loader_settings=loader_settings)
    val_data = VideoDataset(root_dir, mean_image_name, 'val', duration=duration, img_size=img_size, loader_settings=loader_settings)
    test_data = VideoDataset(root_dir, mean_image_name, 'test', duration=duration, img_size=img_size, loader_settings=loader_settings)
    
    mean_image = np.load(os.path.join(root_dir, mean_image_name))
    mean_image = cv2.resize(mean_image, (img_size[1], img_size[0])) # Resize the mean_image to the correct size
    mean_image = mean_image.astype(np.float32)/255. # Convert to [0, 1] (float)
    
    return (train_data, val_data, test_data, mean_image)