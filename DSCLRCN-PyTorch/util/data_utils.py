"""Data utility functions."""
import os
import sys

import numpy as np
import torch
import torch.utils.data as data

from scipy.ndimage import imread
from scipy.misc import imresize

import pickle

import cv2
from tqdm import tqdm

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

    
def get_SALICON_datasets(dataset_folder='Dataset/Transformed/'):
    """
    Load and preprocess the SALICON dataset.
    """

    if not dataset_folder.endswith('/'):
        dataset_folder += '/'

    mean_image = np.load(dataset_folder + 'mean_image.npy').astype(np.float32)/255.
    
    with open(dataset_folder + 'train_datadict.pickle', 'rb') as f:
        train_data = pickle.load(f)
    X_train = [(image.astype(np.float32)/255. - mean_image).transpose(2,0,1) for image in train_data['images']]
    y_train = [fix_map.astype(np.float32)/255. for fix_map in train_data['fix_maps']]
    # Restart the line
    sys.stdout.write('Progress: 50%\r')
    sys.stdout.flush()

    with open(dataset_folder + 'val_datadict.pickle', 'rb') as f:
        val_data = pickle.load(f)
    X_val = [(image.astype(np.float32)/255. - mean_image).transpose(2,0,1) for image in val_data['images']]
    y_val = [fix_map.astype(np.float32)/255. for fix_map in val_data['fix_maps']]
    # Restart the line
    sys.stdout.write('Progress: 75%\r')
    sys.stdout.flush()

    with open(dataset_folder + 'test_datadict.pickle', 'rb') as f:
        test_data = pickle.load(f)
    X_test = [(image.astype(np.float32)/255. - mean_image).transpose(2,0,1) for image in test_data['images']]
    y_test = [fix_map.astype(np.float32)/255. for fix_map in test_data['fix_maps']]
    # Restart the line
    sys.stdout.write('Progress: 100%\r')
    sys.stdout.flush()
    
    return (SaliconData(X_train, y_train),
            SaliconData(X_val, y_val),
            SaliconData(X_test, y_test),
            mean_image)

def get_raw_SALICON_datasets(dataset_folder='C:/Users/simon/Downloads/Project Datasets/SALICON/', height=480, width=640):
    """
    Load and preprocess the SALICON dataset from the raw images and fixation maps.
    """
    def read_raw_dataset(input_directory='C:/Users/simon/Downloads/Project Datasets/SALICON/', height=480, width=640):
        """
        Read the raw images into lists and return them.
        """
        # Loop over each file in the Dataset folder
        if not input_directory.endswith('/'):
            input_directory += '/'
        full_directory = input_directory + 'images'

        # Create the three dictionaries
        train_datadict = {'images': np.empty((10000, height, width, 3), dtype=np.uint8), 'fix_maps': np.empty((10000, height, width), dtype=np.uint8)}
        val_datadict = {'images': np.empty((5000, height, width, 3), dtype=np.uint8), 'fix_maps': np.empty((5000, height, width), dtype=np.uint8)}
        test_datadict = {'images': np.empty((5000, height, width, 3), dtype=np.uint8), 'fix_maps': np.empty((5000, height, width), dtype=np.uint8)}
        train_index = 0
        val_index = 0
        test_index = 0

        print('')
        print('Reading dataset from {}'.format(input_directory))
        
        # Read the files and put their names in a list
        if os.name == 'posix':
            # Unix
            file_names = os.listdir(full_directory)
        else:
            # Windows (os.name == 'nt')
            with os.scandir(full_directory) as file_iterator:
                file_names = [file_object.name for file_object in list(file_iterator)]

        print('')
        print('Reading image files: All')

        # Sort the list
        file_names = sorted(file_names)
        count = 0
        for image_file in tqdm(file_names):
            # Check if the file is .jpg
            if image_file.endswith('.jpg'):
                # read the image
                img = cv2.imread(full_directory + '/' + image_file)
                # convert the image to RGB, and store values as floats in range 0-1
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize the image to width x height (width wide, height high)
                resized_img = cv2.resize(img, (width, height)).astype(np.uint8)
                if 'train' in image_file:
                    # Add this to the train dict
                    train_datadict['images'][train_index] = resized_img
                    train_index += 1
                elif 'test' in image_file:
                    test_datadict['images'][test_index] = resized_img
                    test_index += 1
                elif 'val' in image_file:
                    val_datadict['images'][val_index] = resized_img
                    val_index += 1
                else:
                    # Error
                    print('Error: unexpected file encountered ({})'.format(image_file.name))

        # Do the same for the fixation maps for training data
        print('')
        print('Reading fixation maps: Training')
        full_directory = input_directory + '/fixation maps/train'
        if os.name == 'posix':
            # Unix
            file_names = os.listdir(full_directory)
        else:
            # Windows (os.name == 'nt')
            with os.scandir(full_directory) as file_iterator:
                file_names = [file_object.name for file_object in list(file_iterator)]

        file_names = sorted(file_names)
        count = 0
        for image_file in tqdm(file_names):
            # Check if the file is .png
            if image_file.endswith('.png'):
                # read the image
                img = cv2.imread(full_directory + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                # Resize the image to width x height
                resized_img = cv2.resize(img, (width, height)).astype(np.uint8)
                # Add this to the train dict
                train_datadict['fix_maps'][count - 1] = resized_img

        # Do the same for the fixation maps for validation data
        print('')
        print('Reading fixation maps: Validation')
        full_directory = input_directory + '/fixation maps/val'
        if os.name == 'posix':
            # Unix
            file_names = os.listdir(full_directory)
        else:
            # Windows (os.name == 'nt')
            with os.scandir(full_directory) as file_iterator:
                file_names = [file_object.name for file_object in list(file_iterator)]

        file_names = sorted(file_names)
        count = 0
        for image_file in tqdm(file_names):
            # Check if the file is .png
            if image_file.endswith('.png'):
                # read the image
                img = cv2.imread(full_directory + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                # Resize the image to width x height
                resized_img = cv2.resize(img, (width, height)).astype(np.uint8)
                # Add this to the val dict
                val_datadict['fix_maps'][count - 1] = resized_img

        # Compute the mean train image
        print('Computing mean image')
        mean_image = np.average(train_datadict['images'], axis=0).astype(np.uint8)

        # Return the data dicts and the mean image
        return train_datadict, val_datadict, test_datadict, mean_image

    if not dataset_folder.endswith('/'):
        dataset_folder += '/'

    # Create the initial dictionaries of the raw images and fixation maps
    train_data, val_data, test_data, mean_image = read_raw_dataset(input_directory=dataset_folder, height=height, width=width)

    # Process the training data
    X_train = train_data['images']
    for i, image in enumerate(X_train):
        X_train[i] = (image.astype(np.float32)/255. - mean_image)
    X_train = X_train.transpose(0, 3, 1, 2)
    # Restart the line
    sys.stdout.write('Progress:  25%\r')
    sys.stdout.flush()

    y_train = train_data['fix_maps']
    for i, fix_map in enumerate(y_train):
        y_train[i] = fix_map.astype(np.float32)/255.
    # Restart the line
    sys.stdout.write('Progress:  50%\r')
    sys.stdout.flush()

    # Process the validation data
    X_val = val_data['images']
    for i, image in enumerate(X_val):
        X_val[i] = (image.astype(np.float32)/255. - mean_image)
    X_val = X_val.transpose(0, 3, 1, 2)
    # Restart the line
    sys.stdout.write('Progress:  63%\r')
    sys.stdout.flush()

    y_val = val_data['fix_maps']
    for i, fix_map in enumerate(y_val):
        y_val[i] = fix_map.astype(np.float32)/255.
    # Restart the line
    sys.stdout.write('Progress:  75%\r')
    sys.stdout.flush()

    # Process the testing data
    X_test = test_data['images']
    for i, image in enumerate(X_test):
        X_test[i] = (image.astype(np.float32)/255. - mean_image)
    X_test = X_test.transpose(0, 3, 1, 2)
    # Restart the line
    sys.stdout.write('Progress:  88%\r')
    sys.stdout.flush()

    y_test = test_data['fix_maps']
    for i, fix_map in enumerate(y_test):
        y_test[i] = fix_map.astype(np.float32)/255.
    # Restart the line
    sys.stdout.write('Progress: 100%\r')
    sys.stdout.flush()
    
    return (SaliconData(X_train, y_train),
            SaliconData(X_val, y_val),
            SaliconData(X_test, y_test),
            mean_image)

def get_SALICON_subset(file_name, dataset_folder='Dataset/Transformed/'):
    if not dataset_folder.endswith('/'):
        dataset_folder += '/'

    mean_image = np.load(dataset_folder + 'mean_image.npy').astype(np.float32)/255.

    with open(dataset_folder + ''+file_name, 'rb') as f:
        data = pickle.load(f)
    X = [image.astype(np.float32)/255. - mean_image for image in data['images']]
    y = [fix_map.astype(np.float32)/255. for fix_map in data['fix_maps']]
            
    return SaliconData(X, y)
