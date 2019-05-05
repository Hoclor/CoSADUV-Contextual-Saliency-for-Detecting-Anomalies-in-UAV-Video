# NoTemporal DoM:
# 2.2 frames/s on GeForce RTX 2080 Ti, 10989 MB memory, using 3200 MB GPU memory and 3700 MB RAM, with
import pickle

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import cv2
from models.CoSADUV import CoSADUV
from models.CoSADUV_NoTemporal import CoSADUV_NoTemporal
from models.DSCLRCN import DSCLRCN
from util import loss_functions
from util.data_utils import get_SALICON_datasets, get_video_datasets
from util.solver import Solver

# torch.multiprocessing.set_start_method("forkserver")  # spawn, forkserver, or fork

location = ""  # ncc or '', where the code is to be run (affects output)
if location == "ncc":
    print_func = print
else:
    print_func = tqdm.write

### Data options ###

dataset_name = input("Dataset (UAV123/EyeTrackUAV): ")

if dataset_name in ["UAV123", "EyeTrackUAV"]:
    sequence_name = input("Sequence name: ")

if dataset_name not in ["SALICON", "UAV123", "EyeTrackUAV"]:
    print_func("Error: unrecognized dataset '{}'".format(dataset_name))
    exit()

dataset_root_dir = "Dataset/" + dataset_name  # Dataset/[SALICON, UAV123]
# Name of mean_image file: Must be located at dataset_root_dir/mean_image_name
mean_image_name = "mean_image.npy"
# Height, width of images
img_size = (480, 640)  # Original: 480, 640, reimplementation: 96, 128
# Duration: Length of sequences loaded from each video, if a video dataset is used
duration = 300

### Testing options ###

# Minibatchsize: Determines how many images are processed at a time on the GPU
minibatchsize = 1  # Recommended: 4 for 480x640 for >12GB mem, 2 for <12GB mem.
# Loss functions:
# From loss_functions (use loss_functions.LOSS_FUNCTION_NAME)
# NSS_loss
# CE_MAE_loss
# PCC_loss
# KLDiv_loss
loss_funcs = [loss_functions.NSS_alt, loss_functions.MAE_loss, loss_functions.DoM]  # Recommended: NSS_loss

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
else:
    # Assume dataset is in VideoDataset structure
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

### Testing ###
def test_model(model, test_set, loss_fn, location="ncc"):
    # Set the model to evaluation mode
    model.eval()

    loss = 0
    count = 0
    test_loop = test_set
    if location != "ncc":
        test_loop = tqdm(test_loop, desc="Test (best checkpoint)")
    for video_loader in test_loop:
        if location != "ncc":
            video_loader = tqdm(video_loader, desc="Video")

        # If the model is temporal, reset its temporal state
        # at the start of each video
        if model.temporal:
            model.clear_temporal_state()

        for data in video_loader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # Produce the output
            outputs = model(inputs).squeeze(1)
            # Move the output to the CPU so we can process it using numpy
            outputs = outputs.cpu().data.numpy()

            # Resize the images to input size
            outputs = np.array(
                [
                    cv2.resize(output, (labels.shape[2], labels.shape[1]))
                    for output in outputs
                ]
            )
            # Apply a Gaussian filter to blur the saliency maps
            sigma = 0.035 * min(labels.shape[1], labels.shape[2])
            kernel_size = int(4 * sigma)
            # make sure the kernel size is odd
            kernel_size += 1 if kernel_size % 2 == 0 else 0
            outputs = np.array(
                [
                    cv2.GaussianBlur(output, (kernel_size, kernel_size), sigma)
                    for output in outputs
                ]
            )

            outputs = torch.from_numpy(outputs)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                labels = labels.cuda()
            loss += loss_fn(outputs, labels).item()
            count += 1
    return loss, count


print_func("Testing model")
print_func("(on val set if using SALICON, otherwise on test set)\n")

test_loss, test_count = test_model(
    model, test_loader, test_loss_func, location=location
)

# Delete the model to free up memory
del model
filename = "trained_models/best_model_" + model_name + ".pth"

# Load the checkpoint
if torch.cuda.is_available():
    checkpoint = torch.load(filename)
else:
    checkpoint = torch.load(filename, map_location="cpu")
start_epoch = checkpoint["epoch"]
# Create the model
model = CoSADUV_NoTemporal(input_dim=img_size, local_feats_net="Seg")
model.load_state_dict(checkpoint["state_dict"])
if torch.cuda.is_available():
    model = model.cuda()

# Test the checkpoint
print_func("Testing best checkpoint, after {} epochs of training".format(start_epoch))
test_loss_checkpoint, test_count_checkpoint = test_model(
    model, test_loader, test_loss_func, location=location
)

# Print out the result
print()
print("{} score on test set:".format(test_loss_func.__name__))
print("(Higher is better)")
print("Last model     : {:6f}".format(-1 * test_loss / test_count))
print(
    "Best Checkpoint: {:6f}".format(-1 * test_loss_checkpoint / test_count_checkpoint)
)
