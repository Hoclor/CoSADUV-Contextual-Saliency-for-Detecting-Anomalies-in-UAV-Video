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

def convert_SALICON_Data(input_directory='C:/Users/simon/Downloads/Project Datasets/SALICON',
    output_directory='../DSCLRCN-PyTorch/Dataset/Transformed', height=96, width=128):
    # Converts the images in the SALICON dataset to the format required by
    # the PyTorch implementation of DSCLRCN

    # Loop over each file in the Dataset/Untransformed folder
    full_directory = input_directory + '/images'

    # Create the three dictionaries
    train_datadict = {'images': np.empty((10000, height, width, 3), dtype=np.uint8), 'fix_maps': np.empty((10000, height, width), dtype=np.uint8)}
    val_datadict = {'images': np.empty((5000, height, width, 3), dtype=np.uint8), 'fix_maps': np.empty((5000, height, width), dtype=np.uint8)}
    test_datadict = {'images': np.empty((5000, height, width, 3), dtype=np.uint8), 'fix_maps': np.empty((5000, height, width), dtype=np.uint8)}
    train_index = 0
    val_index = 0
    test_index = 0

    print('')
    print('Reading image files from {}'.format(full_directory))
    
    with os.scandir(full_directory) as file_iterator:
        count = 0
        file_names = [file_object.name for file_object in list(file_iterator)]
        file_names = sorted(file_names)
        for image_file in file_names:
            if count % 200 == 0:
                # Restart the line
                sys.stdout.write('Progress: {:3.0f}%\r'.format(count/200))
                sys.stdout.flush()
            count += 1

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

    # Restart the line
    sys.stdout.write('Progress: 100%\r')
    sys.stdout.flush()
    print('')

    # Do the same for the fixation maps for training data
    print('')
    print('Reading fixation maps: Training')
    full_directory = input_directory + '/fixation maps/train'
    with os.scandir(full_directory) as file_iterator:
        index = 0
        count = 0
        file_names = [file_object.name for file_object in list(file_iterator)]
        file_names = sorted(file_names)
        for image_file in file_names:
            if count % 100 == 0:
                # Restart the line
                sys.stdout.write('Progress: {:3.0f}%\r'.format(count/100))
                sys.stdout.flush()
            count += 1

            # Check if the file is .png
            if image_file.endswith('.png'):
                # read the image
                img = cv2.imread(full_directory + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                # Resize the image to width x height
                resized_img = cv2.resize(img, (width, height)).astype(np.uint8)
                # Add this to the train dict
                train_datadict['fix_maps'][count - 1] = resized_img

    # Restart the line
    sys.stdout.write('Progress: 100%\r')
    sys.stdout.flush()
    print('')

    # Do the same for the fixation maps for validation data
    print('')
    print('Reading fixation maps: Validation')
    full_directory = input_directory + '/fixation maps/val'
    with os.scandir(full_directory) as file_iterator:
        count = 0
        file_names = [file_object.name for file_object in list(file_iterator)]
        file_names = sorted(file_names)
        for image_file in file_names:
            if count % 50 == 0:
                # Restart the line
                sys.stdout.write('Progress: {:3.0f}%\r'.format(count/50))
                sys.stdout.flush()
            count += 1

            # Check if the file is .png
            if image_file.endswith('.png'):
                # read the image
                img = cv2.imread(full_directory + '/' + image_file, cv2.IMREAD_GRAYSCALE)
                # Resize the image to width x height
                resized_img = cv2.resize(img, (width, height)).astype(np.uint8)
                # Add this to the val dict
                val_datadict['fix_maps'][count - 1] = resized_img

    # Restart the line
    sys.stdout.write('Progress: 100%\r')
    sys.stdout.flush()
    print('')

    # Compute the mean train image
    print('Computing mean image')
    mean_image = np.average(train_datadict['images'], axis=0).astype(np.uint8)

    print('Writing Data to {}'.format(output_directory))
    
    print('Writing mean image')
    np.save(output_directory + '/mean_image.npy', mean_image)
    print('Writing Training data')
    with open(output_directory + '/train_datadict.pickle', 'wb') as f:
        pickle.dump(train_datadict, f)
    print('Writing Testing data')
    with open(output_directory + '/test_datadict.pickle', 'wb') as f:
        pickle.dump(test_datadict, f)
    print('Writing Validation data')
    with open(output_directory + '/val_datadict.pickle', 'wb') as f:
        pickle.dump(val_datadict, f)
    print('Done')


convert_SALICON_Data(height=96, width=128)
