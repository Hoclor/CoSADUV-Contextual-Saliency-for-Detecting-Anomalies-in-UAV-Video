# DSCLRCN-PyTorch
PyTorch implementation of https://arxiv.org/abs/1610.01708

- The main file used to run training and testing is 'DSCLRCN_exp.ipynb'. Please have a look at it

- All the pretrained models used to build the architecture of the network are available as PyTorch loadable models under the following GDrive link: https://drive.google.com/open?id=1NiIkHm9e1fO7ilox4ZdRGkgiQZHbcr7x
Please download them and copy them into their corresponding folders

- The dataset should be arranged in the directory 'Dataset/Transformed' as three dictionaries ('train_datadict', 'val_datadict' and test_datadict') each containing the images and fixation maps resized to 92x128; in addition to a mean image computed from the training dataset for normalization. Please have a look at 'util/data_utils.py' file

- The original sources of the models:
PlacesCNN: http://places.csail.mit.edu/downloadCNN.html
Segmentation ResNet50 Encoder: https://github.com/hangzhaomit/semantic-segmentation-pytorch
VGG16 (available with PyTorch 'torchvision' library)
