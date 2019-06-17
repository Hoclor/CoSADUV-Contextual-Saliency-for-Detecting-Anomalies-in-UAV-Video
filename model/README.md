# Contextual Saliency for Detecting Anomalies in UAV Video

Repository for my Master's Project on the topic of "Contextual Saliency for Detecting Anomalies within Unmanned Aerial Vehicle (UAV) Video"

## Structure

The 'data_preprocessing' folder (up one level) contains a set of scripts for processing data and datasets to prepare them for use with the model. 

This folder ('model') contains all files for the model architecture, pre-trained models used, datasets, training, testing, etc:
- 'Dataset' is where the dataset(s) to be used should be placed. Support for the SALICON and UAV123 datasets are built-in, and any other datasets with the same structure as one of these should be straightforward to use. Other datasets may require extra work in creating new dataloaders or preprocessing scripts.
- 'models' contains all files for the models used by CoSADUV (ResNet50, Places-CNN, potentially vgg16, and the convolutional-LSTM layer), and the different CoSADUV models themselves (as well as the original DSCLRCN models). Pre-trained model files for ResNet50 and Places-CNN should be placed inside this folder as specified by the README found in model/
- 'testing' contains a few scripts for testing the model, as well as producing (live) demonstrations of its performance
- 'trained_models' is a folder for placing pretrained models in that can be loaded by the main files
- 'util' contains a set of utility scripts (data loading, loss functions, solver script for training)
- 'main_ncc.py' and 'notebook_main.ipynb' contain the high-level code for running, training, and testing the network.

## Instructions

**To train or run the model, please see 'main_ncc.py' or 'notebook_main.ipynb'.**

- The main files used to run training and testing are 'notebook_main.ipynb' and 'main_ncc.py'. Please have a look at them. They are set up to train and test the model with minor changes required. The IPython notebook file provides a more interactive setup, while 'main_ncc' can be run start to finish without any input required, and can be used if IPython is not available.

- The following pretrained models are used to build the architecture of the network. Please download them and place inside their respective folders inside the 'models' folder:

    - Places-CNN: available under https://drive.google.com/open?id=1NiIkHm9e1fO7ilox4ZdRGkgiQZHbcr7x (not maintained by the author, provided as part of the repository linked above)
    - ResNet50: available at https://www.kaggle.com/pytorch/resnet50

- The dataset (SALICON or UAV123) should be arranged inside the 'Dataset' folder as specified in the README found there. Other datasets are not directly supported, and may require additional work to use.

- Hyperparameters and options such as dataset, learning rate, epoch numbers, and more are adjustable in the main files.

Initial/base code in this folder cloned from https://github.com/AAshqar/DSCLRCN-PyTorch (PyTorch implementation of https://arxiv.org/abs/1610.01708)

The original sources of the models:\
PlacesCNN: http://places.csail.mit.edu/downloadCNN.html \
Segmentation ResNet50 Encoder: https://github.com/hangzhaomit/semantic-segmentation-pytorch \
