import math
from random import shuffle
import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from random import *

from tqdm import tqdm
from tqdm import tqdm_notebook

from util import data_utils


def prepare_parameters(model, optim_args):
    """Produce two groups of parameters:
        - the pretrained parameters (PlacesCNN, ResNet50), and
        - the other parameters (not pretrained)
    This allows us to set the lr of the two groups separately
    (other params as the lr given as input, pretrained params as this lr * 0.1)
    """
    pretrained_parameters = [
        param
        for name, param in model.named_parameters()
        if (name.startswith("local_feats") and not ".bn" in name)
        or name.startswith("context")
    ]
    new_parameters = [
        param
        for name, param in model.named_parameters()
        if not (name.startswith("local_feats") or name.startswith("context"))
    ]
    pretrained_param_group = optim_args.copy()
    pretrained_param_group["lr"] *= 1e-1
    pretrained_param_group["params"] = pretrained_parameters

    # Fix the scale (weight) and bias params of the BN layers in the local_feats model
    for name, param in model.named_parameters():
        if ".bn" in name:
            param.requires_grad = False

    return pretrained_param_group, new_parameters


def get_time_format(time_in_seconds):
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(time_in_seconds % 60)
    # Convert to two digit format, as string
    hours = "0" + str(hours) if hours < 10 else str(hours)
    minutes = "0" + str(minutes) if minutes < 10 else str(minutes)
    seconds = "0" + str(seconds) if seconds < 10 else str(seconds)
    return hours, minutes, seconds


class Solver(object):
    default_adam_args = {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    default_SGD_args = {"lr": 1e-2, "weight_decay": 0.0005, "momentum": 0.9}

    def __init__(
        self,
        optim=torch.optim.Adam,
        optim_args={},
        loss_func=torch.nn.KLDivLoss(),
        location="ncc",
    ):
        if optim == torch.optim.Adam:
            optim_args_merged = self.default_adam_args.copy()
        else:
            optim_args_merged = self.default_SGD_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

        self.location = location

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = 1

    def train(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        num_minibatches=1,
        log_nth=0,
        filename_args={},
    ):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data, list of torch.utils.data.DataLoader objects
        - val_loader: validation data, list of torch.utils.data.DataLoader objects
        - num_epochs: total number of training epochs
        - num_minibatches: the number of minibatches per bath
        - log_nth: log training accuracy and loss every nth iteration
        - filename_args: parameters for naming the checkpoint file
        """
        ### Prepare optimiser ###

        # Move model to cuda first, if available, so optimiser is initialized properly
        if torch.cuda.is_available():
            model.cuda()
        # Reducing lr of pretrained parameters and freeze batch-norm weights
        pretrained_parameters, new_parameters = prepare_parameters(
            model, self.optim_args
        )
        optim = self.optim(new_parameters, **self.optim_args)
        optim.add_param_group(pretrained_parameters)
        self._reset_histories()
        # Create the scheduler to allow lr adjustment
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.4)

        ### Training ###

        # Sum up the length of each loader in train_loader
        iter_per_epoch = int(
            math.ceil(sum([len(loader) for loader in train_loader]) / num_minibatches)
        )  # Count an iter as a full batch, not a minibatch

        tqdm.write("START TRAIN.")

        nIterations = num_epochs * iter_per_epoch

        tqdm.write("")
        tqdm.write("Number of epochs: {}".format(num_epochs))
        tqdm.write(
            "Approx. train frames per epoch: {}".format(
                iter_per_epoch * filename_args["batchsize"]
            )
        )
        tqdm.write(
            "Approx. val frames per epoch: {}".format(
                iter_per_epoch * filename_args["batchsize"]
            )
        )
        tqdm.write("Frames per batch: {}".format(filename_args["batchsize"]))
        tqdm.write(
            "Number of iterations/batches per (train) epoch: {}".format(iter_per_epoch)
        )
        tqdm.write("Train accuracy recorded every {} iterations".format(log_nth))
        tqdm.write("")

        epoch_loop = range(num_epochs)
        if self.location != "ncc":
            if self.location == "jupyter":
                epoch_loop = tqdm_notebook(epoch_loop, desc="Model (train+val)")
            else:
                epoch_loop = tqdm(epoch_loop, desc="Model (train+val)")

        # Iteration counter of batches (NOT minibatches)
        it = 0

        total_time = 0
        # Epoch
        for j in epoch_loop:
            start_time = time.time()
            train_loss_logs = 0
            # Downscale the learning rate by a factor of 2.5 every epoch
            scheduler.step()

            # Set the model to training mode
            model.train()

            if self.location == "ncc":
                outer_train_loop = enumerate(train_loader, 0)
            elif self.location == "jupyter":
                outer_train_loop = enumerate(
                    tqdm_notebook(train_loader, desc="Epoch (train)"), 0
                )
            else:
                outer_train_loop = enumerate(
                    tqdm(train_loader, desc="Epoch (train)"), 0
                )

            counter = 0  # counter for minibatches
            it = j * iter_per_epoch

            # Repeat training for each loader in the train_loader
            for k, loader in outer_train_loop:
                if self.location == "ncc":
                    inner_train_loop = enumerate(loader, 0)
                elif self.location == "jupyter":
                    inner_train_loop = enumerate(tqdm_notebook(loader, desc="Video"), 0)
                else:
                    inner_train_loop = enumerate(tqdm(loader, desc="Video"), 0)

                # Repeat training for each batch in the loader
                for i, data in inner_train_loop:
                    # Batch of items in training set

                    # Count the number of minibatches performed since last backprop
                    counter += 1

                    # Load the items and labels in this batch from the train_loader
                    inputs, labels = data
                    # Unsqueeze labels so they're shaped as [10, 96, 128, 1]
                    labels = labels.unsqueeze(3)

                    # Convert these to cuda if cuda is available
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()

                    inputs = Variable(inputs)
                    labels = Variable(labels)

                    # train the model (forward propagation) on the inputs
                    outputs = model(inputs)
                    # transpose the outputs so it's in the order [N, H, W, C]
                    # instead of [N, C, H, W]
                    outputs = outputs.transpose(1, 3)
                    outputs = outputs.transpose(1, 2)

                    loss = self.loss_func(outputs, labels)
                    loss.backward()
                    # Only step and zero the gradients every num_minibatches steps
                    if counter == num_minibatches:
                        counter = 0  # Reset the minibatch counter
                        optim.step()
                        optim.zero_grad()
                        # Print results every log_nth batches,
                        # or if this is the last batch of the last loader
                        if it % log_nth == 0 or (
                            (i == len(loader) - 1) and (k == len(train_loader) - 1)
                        ):
                            tqdm.write(
                                "[Iteration %i/%i] TRAIN loss: %f"
                                % (it, nIterations, loss)
                            )
                            self.train_loss_history.append(loss.item())
                            train_loss_logs += 1
                        it += 1  # iteration (batch) number

                    # Free up memory
                    del inputs, outputs, labels, loss

            # Free up memory
            del outer_train_loop, inner_train_loop, data, loader

            ### Validation ###
            model.eval()

            if self.location == "ncc":
                outer_val_loop = enumerate(val_loader, 0)
            elif self.location == "jupyter":
                outer_val_loop = enumerate(
                    tqdm_notebook(val_loader, desc="Validation"), 0
                )
            else:
                outer_val_loop = enumerate(tqdm(val_loader, desc="Validation"), 0)

            val_loss = 0
            # Repeat validation for each loader in outer_val_loop
            for kk, loader in outer_val_loop:
                if self.location == "ncc":
                    inner_val_loop = enumerate(loader, 0)
                elif self.location == "jupyter":
                    inner_val_loop = enumerate(tqdm_notebook(loader, desc="Video"), 0)
                else:
                    inner_val_loop = enumerate(tqdm(loader, desc="Video"), 0)

                for ii, data in inner_val_loop:
                    inputs, labels = data
                    # Unsqueeze labels so they're shaped as [batch_size, H, W, 1]
                    labels = labels.unsqueeze(3)

                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs_val = Variable(inputs)
                    labels_val = Variable(labels)

                    outputs_val = model(inputs_val)
                    # transpose the outputs so it's in the order [N, H, W, C]
                    # instead of [N, C, H, W]
                    outputs_val = outputs_val.transpose(1, 3)
                    outputs_val = outputs_val.transpose(1, 2)

                    val_loss += self.loss_func(outputs_val, labels_val).item()

                    # Free up memory
                    del inputs_val, outputs_val, labels_val, inputs, labels

            # Free up memory
            del outer_val_loop, inner_val_loop, data, loader

            # Compute avg loss
            val_loss /= sum([len(vloader) for vloader in val_loader])

            self.val_loss_history.append(val_loss)
            # Check if this is the best validation loss so far.
            # If so, save the current model state
            if val_loss < self.best_val_loss:
                if len(filename_args) < 3:
                    filename = "trained_models/model_state_dict_best_loss_{:6f}.pth".format(
                        val_loss
                    )
                else:
                    filename = "trained_models/best_model_{}_lr2_batch{}_epoch{}.pth".format(
                        filename_args["optim"],
                        filename_args["batchsize"],
                        filename_args["epoch_number"],
                    )
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": j + 1,
                        "state_dict": model.state_dict(),
                        "best_accuracy": val_loss,
                    },
                    filename,
                )
                tqdm.write("Checkpoint created with loss: {:6f}".format(val_loss))

            # Free up memory
            del val_loss

            # Compute the time taken for this epoch and in total
            this_time = time.time() - start_time
            total_time += this_time
            # Print this info out in hh:mm:ss format
            hours_this, minutes_this, seconds_this = get_time_format(this_time)
            hours_total, minutes_total, seconds_total = get_time_format(total_time)

            # Print the average Train loss for the last epoch
            # (avg of the logged losses, as decided by log_nth value)
            tqdm.write(
                "[Epoch %i/%i] TRAIN NSS Loss: %f"
                % (
                    j,
                    num_epochs,
                    sum(self.train_loss_history[-train_loss_logs:]) / train_loss_logs,
                )
            )
            tqdm.write(
                "[Epoch %i/%i] VAL NSS Loss: %f"
                % (j, num_epochs, self.val_loss_history[-1])
            )
            tqdm.write(
                "Time taken: {}:{}:{} last epoch, {}:{}:{} total".format(
                    hours_this,
                    minutes_this,
                    seconds_this,
                    hours_total,
                    minutes_total,
                    seconds_total,
                )
            )

        tqdm.write("FINISH.")

