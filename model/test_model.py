import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import pickle
from random import randint, randrange
import sys
from tqdm import tqdm
import cv2
print("CUDA available: {}".format(torch.cuda.is_available()))

# Import model architectures
from models.DSCLRCN_OldContext import DSCLRCN
from models.CoSADUV import CoSADUV
from models.CoSADUV_NoTemporal import CoSADUV_NoTemporal

# Prepare settings and get the datasets
from util.data_utils import get_SALICON_datasets, get_video_datasets

### Data options ###
dataset_root_dir = "Dataset/UAV123"  # Dataset/[SALICON, UAV123]
mean_image_name = (
    "mean_image.npy"
)  # Must be located at dataset_root_dir/mean_image_name
img_size = (480, 640)  # height, width - original: 480, 640, reimplementation: 96, 128
duration = 300  # Length of sequences loaded from each video, if a video dataset is used

from util import loss_functions

from util.solver import Solver

### Testing options ###

# Minibatchsize: Determines how many images are processed at a time on the GPU
minibatchsize = 1  # Recommended: 4 for 480x640 for >12GB mem, 2 for <12GB mem.

# Loss functions:
# From loss_functions (use loss_functions.LOSS_FUNCTION_NAME)
# NSS_loss
# CE_MAE_loss
# PCC_loss
# KLDiv_loss
loss_func = loss_functions.NSS_alt  # Recommended: NSS_loss
test_loss_func = loss_functions.CE_MAE_loss

########## PREPARE DATASETS ##########

### Prepare datasets and loaders ###

if "SALICON" in dataset_root_dir:
    train_data, val_data, test_data, mean_image = get_SALICON_datasets(
        dataset_root_dir, mean_image_name, img_size
    )
    train_loader = [
        torch.utils.data.DataLoader(
            train_data,
            batch_size=minibatchsize,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    ]
    val_loader = [
        torch.utils.data.DataLoader(
            val_data,
            batch_size=minibatchsize,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    ]
    # Load test loader using val_data as SALICON does not give GT for its test set
    test_loader = [
        torch.utils.data.DataLoader(
            val_data,
            batch_size=minibatchsize,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    ]
elif "UAV123" in dataset_root_dir:
    train_loader, val_loader, test_loader, mean_image = get_video_datasets(
        dataset_root_dir,
        mean_image_name,
        duration=duration,
        img_size=img_size,
        shuffle=False,
        loader_settings={
            "batch_size": minibatchsize,
            "num_workers": 8,
            "pin_memory": False,
        },
    )

########## LOADING MODELS ##########

# Loading a model from the saved state that produced
# the lowest validation loss during training:

# Requires the model classes be loaded

# Assumes the model uses models.CoSADUV_NoTemporal architecture.
# If not, this method will fail
def load_model_from_checkpoint(model_name):
    filename = "trained_models/" + model_name + ".pth"
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location="cpu")
    start_epoch = checkpoint["epoch"]
    best_accuracy = checkpoint["best_accuracy"]
    
    if "DSCLRCN" in model_name:
        model = DSCLRCN(input_dim=img_size, local_feats_net="Seg")
    elif "CoSADUV_NoTemporal" in model_name:
        model = CoSADUV_NoTemporal(input_dim=img_size, local_feats_net="Seg")
    elif "CoSADUV" in model_name:
        model = CoSADUV(input_dim=img_size, local_feats_net="Seg")
    else:
        print("Error: no model name found in filename: {}".format(model_name))
        return
    # Ignore extra parameters ('.num_batches_tracked'
    # that are added on NCC due to different pytorch version)
    model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )

    print(
        "=> loaded model checkpoint '{}' (trained for {} epochs)\n   with architecture {}".format(
            model_name, checkpoint["epoch"], type(model).__name__
        )
    )

    if torch.cuda.is_available():
        model = model.cuda()
        print("   loaded to cuda")
    model.eval()
    return model


def load_model(model_name):
    model = torch.load("trained_models/" + model_name, map_location="cpu")
    print("=> loaded model '{}'".format(model_name))
    if torch.cuda.is_available():
        model = model.cuda()
        print("   loaded to cuda")
    model.eval()
    return model

########## LOAD THE MODELS ##########
models = []
model_names = []
# Loading some pretrained models to test them on the images:

# DSCLRCN models
## Trained on SALICON
### NSS_loss
# model_names.append("DSCLRCN/SALICON/NSS -1.62NSS val best and last/best_model_DSCLRCN_NSS_loss_batch20_epoch5")
## Trained on UAV123
### NSS_alt loss func
# model_names.append("DSCLRCN/UAV123/NSS_alt 1.38last 3.15best testing/best_model_DSCLRCN_NSS_alt_batch20_epoch5")


# CoSADUV_NoTemporal models
## Trained on UAV123
### DoM loss func
# NEED TO COLLECT
### NSS_alt loss func
# RUNNING ON COMP #4 E216
### CE_MAE loss func
# model_names.append("CoSADUV_NoTemporal/best_model_CoSADUV_NoTemporal_CE_MAE_loss_batch20_epoch10")


# CoSADUV models (CoSADUV2)
## Trained on UAV123
### DoM loss func
### NSS_alt loss func
model_names.append("CoSADUV/Adam lr 1e-3 1frame backprop size1 kernel -2train -0.7val 1epoch/best_model_CoSADUV_NSS_alt_batch20_epoch5")
### CE_MAE loss func

max_name_len = max([len(name) for name in model_names])
# Load the models specified above
iterable = model_names

for i, name in enumerate(iterable):
    if "best_model" in name:
        models.append(load_model_from_checkpoint(name))
    else:
        models.append(load_model(name))

print()
print("Loaded all specified models")

########## TEST THE MODEL ##########

# Define a function for testing a model
# Output is resized to the size of the data_source
def test_model(model, data_loader, loss_fn=loss_functions.MAE_loss):
    loss_sum = 0
    loss_sum_2 = 0 # Only used for NSS_alt
    loss_count = 0
    loss_count_2 = 0 # Only used for NSS_alt
    for video_loader in tqdm(data_loader):
        # Reset temporal state if model is temporal
        if model.temporal:
            model.clear_temporal_state()
        for data in tqdm(video_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # Produce the output
            outputs = model(inputs).squeeze(1)
            # if model is temporal detach its state
            if model.temporal:
                model.detach_temporal_state()
            # Move the output to the CPU so we can process it using numpy
            outputs = outputs.cpu().data.numpy()

            # If outputs contains a single image, insert
            # a singleton batchsize dimension at index 0
            if len(outputs.shape) == 2:
                outputs = np.expand_dims(outputs, 0)

            # Resize the images to input size
            outputs = np.array(
                [
                    cv2.resize(output, (labels.shape[2], labels.shape[1]))
                    for output in outputs
                ]
            )

            outputs = torch.from_numpy(outputs)

            if torch.cuda.is_available():
                outputs = outputs.cuda()
                labels = labels.cuda()
            
            # If loss fn is NSS_alt, manually add std_dev() if the target is all-0
            if loss_fn == loss_functions.NSS_alt:
                for i in range(len(labels)):
                    if labels[i].sum() == 0:
                        loss_sum_2 += outputs[i].std().item()
                        loss_count_2 += 1
                    else:
                        loss_sum += loss_fn(outputs[i], labels[i]).item()
                        loss_count += 1
            else:
                loss_sum += loss_fn(outputs, labels).item()
                loss_count += 1

    return loss_sum, loss_count, loss_sum_2, loss_count_2

# Obtaining NSS Loss values on the test set for different models:
for i, model in enumerate(tqdm(models)):
    for loss_fn in tqdm([loss_functions.NSS_alt, loss_functions.CE_MAE_loss, loss_functions.CE_loss, loss_functions.MAE_loss, loss_functions.DoM]):
        test_losses = []
        test_losses.append([loss_fn, test_model(model, val_loader, loss_fn=loss_fn)])

    # Print out the result
    # Data is stored as [loss_fn, [sum1, count1, sum2, count2]],
    # where sum2 and count2 are only used for NSS_alt
    
    print("[{}] Model: ".format(i, model_names[i]))
    print("{} score on test set:".format(loss_fn.__name__))

    for i, data in enumerate(test_losses):
        sum1, count1, sum2, count2 = data[1]
        if data[0] == loss_functions.NSS_alt:
            tqdm.write(
                ("{:25} : {:6f}").format(
                    'NSS_alt (+ve imgs)', sum1 / count1
                )
            )
            tqdm.write(
                ("{:25} : {:6f}").format(
                    'NSS_alt (-ve imgs)', sum2 / count2
                )
            )
        else:
            tqdm.write(
                ("{:25} : {:6f}").format(
                    data[0].__name__, sum1 / count1
                )
            )