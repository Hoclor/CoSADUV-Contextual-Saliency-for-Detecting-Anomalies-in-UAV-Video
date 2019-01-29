'''File containing data-related utility functions for reading/writing data.
'''
# Author = Simon Gokstorp
# Date = November 2018

import os
import sys
import pickle
import numpy as np
import cv2
import re
from tqdm import tqdm

def convert_SALICON_Data(input_directory='C:/Users/simon/Downloads/Project Datasets/SALICON',
    output_directory='../DSCLRCN-PyTorch/Dataset/Transformed', height=96, width=128):
    # Converts the images in the SALICON dataset to the format required by
    # the PyTorch implementation of DSCLRCN

    # Create the mean_image
    mean_image = np.zeros((height, width, 3), dtype=np.uint64)
    count = 0

    # Loop over each file in the Dataset/Untransformed folder
    if not input_directory.endswith('/'):
        input_directory += '/'
    full_directory = input_directory + 'images'

    print('')
    print('Reading dataset from {}'.format(input_directory))
    
    # Read the files and put their names in a list
    if os.name == 'posix':
        # Unix
        file_names = os.listdir(full_directory)
        file_names = [name for name in file_names if 'train' in name]
    else:
        # Windows (os.name == 'nt')
        with os.scandir(full_directory) as file_iterator:
            file_names = [file_object.name for file_object in list(file_iterator) if 'train' in file_object.name]

    print('')
    print('Reading image files: All')

    # Sort the list
    file_names = sorted(file_names)
    for image_file in tqdm(file_names):
        # Check that the file is .jpg and that it's a train image
        if image_file.endswith('.jpg'):
            # read the image
            img = cv2.imread(full_directory + '/' + image_file)
            # convert the image to RGB, and store values as floats in range 0-1
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize the image to width x height (width wide, height high)
            resized_img = cv2.resize(img, (width, height)).astype(np.uint8)
            # Add this to mean_image
            mean_image += resized_img
            count += 1

    # Compute the mean train image
    print('Computing mean image')
    mean_image = (mean_image/count).astype(np.uint8)

    print('Writing mean image')

    try:
        np.save(output_directory + '/mean_image.npy', mean_image)
    except IOError:
        # The directory does not exist, so create it and try writing again
        os.mkdir(output_directory)
        np.save(output_directory + '/mean_image.npy', mean_image)

# Execute this function with h=480, w=640 if this file is the main file, and output_directory renamed to reflect this width/height
if __name__ == "__main__":
    convert_SALICON_Data(input_directory = '../DSCLRCN-PyTorch/Dataset/Raw Dataset/', output_directory='../DSCLRCN-PyTorch/Dataset/Raw Dataset', height=480, width=640)
