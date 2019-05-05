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
def test_model(model, test_set, sequence_name="", loss_fns=[], location="ncc"):
    # Set the model to evaluation mode
    model.eval()
    losses = [0 for _ in loss_fns]
    counts = [0 for _ in loss_fns]
    test_loop = test_set
    # Only use tqdm here if we loop through all videos
    if location != "ncc" and sequence_name == "":
        test_loop = tqdm(test_loop)
    for video_loader in test_loop:
        # Skip videos until the one named sequence_name is found
        if sequence_name != "" and sequence_name != video_loader.video_name:
            continue
        if location != "ncc":
            video_loader = tqdm(video_loader, desc=video_loader.video_name)

        # If the model is temporal, reset its temporal state
        # at the start of each video
        if model.temporal:
            model.clear_temporal_state()

        for i, data in enumerate(video_loader):
            # Reset the model for each image if dataset is SALICON
            if dataset_name == "SALICON" and model.temporal:
                model.clear_temporal_state()
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # Produce the output
            outputs = model(inputs).squeeze(1)
            if inputs.shape != labels.shape:
            # Move the output to the CPU so we can process it using numpy
            outputs = outputs.cpu().data.numpy()
                # Resize the images to labels size
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

            vid_loss = [loss_fn(outputs, labels).item() for loss_fn in loss_fns]
            print_func("Frame [{}]".format(i))
            for j in range(len(vid_loss)):
                losses[j] += vid_loss[i]
                counts[j] += 1
                print_func("    {}: {}".format(loss_fns[i].__name__, vid_loss[i]))
            # Show the input, GT, prediction with cv2
            output = cv2.cvtColor(outputs.squeeze(), cv2.COLOR_GRAY2BGR)
            label = cv2.cvtColor(label.squeeze(), cv2.COLOR_GRAY2BGR)
            _input = cv2.cvtColor(inputs.squeeze(), cv2.COLOR_RGB2BGR)
            out = np.hstack((_input, label, output)) 
            cv2.imshow("Output")
            if cv2.waitKey(1) == 'q':
                cv2.destroyAllWindows()
                return losses, counts
    return losses, counts

if dataset_name == "SALICON":
    print_func("Testing model on SALICON")

test_losses, test_counts = test_model(
    model, test_loader, sequence_name, loss_funcs, location=location
)

# Print out the result
print("Mean scores")
for loss_fn in loss_funcs:
    if test_counts[i] > 0:
        print("{}: {}".format(loss_fn.__name__, test_losses[i] / test_counts[i]))
