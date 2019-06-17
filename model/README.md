# CoSADUV
Initial/base code in this folder cloned from https://github.com/AAshqar/DSCLRCN-PyTorch (PyTorch implementation of https://arxiv.org/abs/1610.01708)

- The main files used to run training and testing are 'notebook_main.ipynb' and 'main_ncc.py'. Please have a look at them

- The following pretrained models are used to build the architecture of the network. Please download them and place inside their respective folders inside the 'models' folder:
    - Places-CNN: available under https://drive.google.com/open?id=1NiIkHm9e1fO7ilox4ZdRGkgiQZHbcr7x (not maintained by the author, provided as part of the repository linked above)
    - ResNet50: available at https://www.kaggle.com/pytorch/resnet50

- The dataset (SALICON or UAV123) should be arranged inside the 'Dataset' folder as specified in the README found there. Other datasets are not directly supported, and may require additional work to use.

The original sources of the models:\
PlacesCNN: http://places.csail.mit.edu/downloadCNN.html \
Segmentation ResNet50 Encoder: https://github.com/hangzhaomit/semantic-segmentation-pytorch \
