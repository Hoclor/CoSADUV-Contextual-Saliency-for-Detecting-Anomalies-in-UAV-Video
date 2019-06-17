# Contextual Saliency for Detecting Anomalies in UAV Video

Repository for my Master's Project on the topic of "Contextual Saliency for Detecting Anomalies within Unmanned Aerial Vehicle (UAV) Video"

## Structure

- 'data_preprocessing' contains a set of scripts for processing data and datasets to prepare them for use with the model. 
- 'model' contains all files for the model architecture, pre-trained models used, datasets, training, testing, etc.
    - 'Dataset' is where the dataset(s) to be used should be placed. Support for the SALICON and UAV123 datasets are built-in, and any other datasets with the same structure as one of these should be straightforward to use. Other datasets may require extra work in creating new dataloaders or preprocessing scripts.
    - 'models' contains all files for the models used by CoSADUV (ResNet50, Places-CNN, potentially vgg16, and the convolutional-LSTM layer), and the different CoSADUV models themselves (as well as the original DSCLRCN models). Pre-trained model files for ResNet50 and Places-CNN should be placed inside this folder as specified by the README found in model/
    - 'testing' contains a few scripts for testing the model, as well as producing (live) demonstrations of its performance
    - 'trained_models' is a folder for placing pretrained models in that can be loaded by the main files
    - 'util' contains a set of utility scripts (data loading, loss functions, solver script for training)
    - **'main_ncc.py' and 'notebook_main.ipynb' are the main files used.** Please have a look at them.

## Instructions

To train or run the model, please see 'main_ncc.py' or 'notebook_main.ipynb'. These files are set up to train and test the model with minor changes required. The IPython notebook file provides a more interactive setup, while 'main_ncc' can be run without interaction, and can be used if IPython is not available. Hyperparameters and options such as dataset, learning rate, epoch numbers, and more are adjustable in these files.