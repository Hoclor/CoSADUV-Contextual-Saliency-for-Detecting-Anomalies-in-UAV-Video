from random import shuffle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from random import *

from tqdm import tqdm
from tqdm import tqdm_notebook


def prepare_parameters(model, optim_args):
    """Produce two groups of parameters: the pretrained parameters (PlacesCNN, ResNet50), and the other parameters
    This allows us to set the lr of the two groups separately (other params as the lr given as input, pretrained params as this lr * 0.1)
    """
    pretrained_parameters = [param for name,param in model.named_parameters() if (name.startswith('local_feats') and not '.bn' in name) or name.startswith('context')]
    new_parameters = [param for name,param in model.named_parameters() if not (name.startswith('local_feats') or name.startswith('context'))]
    pretrained_param_group = optim_args.copy()
    pretrained_param_group['lr'] *= 1e-1
    pretrained_param_group['params'] = pretrained_parameters
    
    # Fix the scale (weight) and bias parameters of the BN layers in the ResNet50 (local_feats) model
    for name, param in model.named_parameters():
        if '.bn' in name:
            param.requires_grad = False

    return pretrained_param_group, new_parameters


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    default_SGD_args = {"lr": 1e-2,
                        "weight_decay": 0.0005,
                        "momentum": 0.9}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.KLDivLoss(), location='ncc',
                 minibatches=1, mean_image=None):
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
        self.minibatches = minibatches
        self.mean_image = mean_image

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = 1

    def train(self, model, train_loader, val_loader, num_epochs=10, num_minibatches=1, log_nth=0, filename_args={}):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in a torch.utils.data.DataLoader based dataloader, or a tuple of data loaders
        - val_loader: val data in a torch.utils.data.DataLoader based dataloader, or a tuple of data loaders
        - num_epochs: total number of training epochs
        - num_minibatches: the number of minibatches to be performed per batch
        - log_nth: log training accuracy and loss every nth iteration
        - filename_args: dict of args required to accurately name the checkpoint
        """
        # Check if a mean_image was supplied to the Solver class.
        # If so, use this to manually normalize each image as they're loaded
        mean_image_given = (type(self.mean_image) != type(None))
        if mean_image_given:
            mean_image = self.mean_image
            mean_image = mean_image.unsqueeze(0) # Add a batchsize dimension
            # mean_image now of shape [1, C, H, W]
        
        # Move the model to cuda first, if applicable, so optimiser is initialized properly
        if torch.cuda.is_available():
            model.cuda()
            if mean_image_given:
                mean_image = mean_image.cuda()

        # Check if train_loader is supplied as a tuple. If it is, we have to load the data differently
        # (The only supported case is when nvvl is used as the data loader)
        if type(train_loader) == tuple:
            nvvl_loader = True
            iter_per_epoch = int(len(train_loader[0])/num_minibatches) # Count an iter as a full batch, not a minibatch
            # Save the tuple as a separate variable to prevent confusion
            train_loader_tuple = train_loader
            del train_loader # Free up memory
            val_loader_tuple = val_loader
            del val_loader # Free up memory
        else:
            nvvl_loader = False
            iter_per_epoch = int(len(train_loader)/num_minibatches) # Count an iter as a full batch, not a minibatch

        # Prepare parameters by reducing lr of pretrained parameters, freezing batch-norm weights
        pretrained_parameters, new_parameters = prepare_parameters(model, self.optim_args)
        optim = self.optim(new_parameters, **self.optim_args)
        optim.add_param_group(pretrained_parameters)
        self._reset_histories()

        # Create the scheduler to allow lr adjustment
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.4)

        tqdm.write('START TRAIN.')
        
        nIterations = num_epochs*iter_per_epoch
        
        epoch_loop = range(num_epochs)
        if self.location != 'ncc':
            if self.location == 'jupyter':
                epoch_loop = tqdm_notebook(epoch_loop)
            else:
                epoch_loop = tqdm(epoch_loop)

        # Iteration counter of batches (NOT minibatches)
        it = 0

        # Epoch
        for j in epoch_loop:
            train_loss_logs = 0
            # Downscale the learning rate by a factor of 2.5 (i.e. multiply by 1/2.5) every epoch
            scheduler.step()

            # Set the model to training mode
            model.train()

            if nvvl_loader:
                # Create the train_loader by zipping the input_data loader (index 0) and the targets loader (index 1)
                train_loader = zip(train_loader_tuple[0], train_loader_tuple[1])
                val_loader = zip(val_loader_tuple[0], val_loader_tuple[1])

            if self.location == 'ncc':
                train_loop = enumerate(train_loader, 0)
            elif self.location == 'jupyter':
                train_loop = enumerate(tqdm_notebook(train_loader), 0)
            else:
                train_loop = enumerate(tqdm(train_loader), 0)
                
            counter = 0 # counter for minibatches
            it = j*iter_per_epoch

            # Batch of items in training set
            for i, data in train_loop:
                counter += 1 # Count the number of minibatches performed since last backprop

                # Load the items in this batch and their labels from the train_loader
                if nvvl_loader:
                    # input is in data[0]['input'], of shape [N, Seq_len, C, H, W]
                    # Since this is the non-temporal model version, only seq_len 1 is supported
                    # Thus, take only the first frame of the sequence
                    inputs = data[0]['input'][:, 0, :, :, :] # shape [N, 1, C, H, W]
                    inputs = inputs.squeeze(1) # shape [N, C, H, W]
                    # labels are in data[1]['input'], of shape [N, Seq_len, C, H, W]
                    # (labels are greyscale but read as RGB)
                    labels = data[1]['input'][:, 0, 0, :, :] # shape [N, 1, 1, H, W]
                    labels = labels.squeeze(1).squeeze(1) # shape [N, H, W]
                else:
                    # input and labels are in data as a tuple
                    inputs, labels = data
                    # inputs shape [N, C, H, W]
                    # labels shape [N, H, W]

                # Free up memory
                del data

                # Normalize inputs by subtracting the mean image, if it was given
                # (this is handled in dataset in torch datasets,
                # but must be done manually here for NVVL datasets)
                if mean_image_given:
                    if inputs.shape != mean_image.shape:
                        mean_image = mean_image[0, :, :].unsqueeze(0)
                        mean_image = mean_image.expand(inputs.shape[0], 3, -1, -1)
                        # mean_image will now be shape [N, C, H, W] for the appropriate N for this batch
                    inputs = inputs - mean_image

                # Convert these to cuda types if cuda is available
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                # DEPRECATED - calling Variable should no longer be necessary, but leave in for now
                inputs = Variable(inputs)
                labels = Variable(labels)
                
                # train the model (forward propagation) on the inputs
                outputs = model(inputs)
                # Squeeze the outputs so it has shape [N, H, W] instead of [N, 1, H, W]
                outputs = outputs.squeeze(1)

                # Free up memory
                del inputs

                loss = self.loss_func(outputs, labels)
                loss.backward()
                # Only step and zero the gradients every num_minibatches steps
                if counter == num_minibatches:
                    counter = 0 # Reset the minibatch counter
                    optim.step()
                    optim.zero_grad()
                    if it%log_nth==0:
                        tqdm.write('[Iteration %i/%i] TRAIN loss: %f' % (it, nIterations, loss))
                        self.train_loss_history.append(loss.item())
                        train_loss_logs += 1
                    it += 1 # iteration (batch) number
                
                # Free up memory
                del outputs, labels, loss
            
            model.eval()
            
            if self.location == 'ncc':
                val_loop = enumerate(val_loader, 0)
            elif self.location == 'jupyter':
                val_loop = enumerate(tqdm_notebook(val_loader), 0)
            else:
                val_loop = enumerate(tqdm(val_loader), 0)
            
            # Validation
            val_loss = 0
            for ii, data in val_loop:
                # Load the items in this batch and their labels from the train_loader
                if nvvl_loader:
                    # input is in data[0]['input'], of shape [N, Seq_len, C, H, W]
                    # Since this is the non-temporal model version, only seq_len 1 is supported
                    # Thus, take only the first frame of the sequence
                    inputs_val = data[0]['input'][:, 0, :, :, :] # shape [N, 1, C, H, W]
                    inputs_val = inputs_val.squeeze(1) # shape [N, C, H, W]
                    # labels are in data[1]['input'], of shape [N, Seq_len, C, H, W]
                    # (labels are greyscale but read as RGB)
                    labels_val = data[1]['input'][:, 0, 0, :, :] # shape [N, 1, 1, H, W]
                    labels_val = labels.squeeze(1).squeeze(1) # shape [N, H, W]
                else:
                    # inputs and labels are in data as a tuple
                    inputs_val, labels_val = data
                    # inputs_val shape [N, C, H, W]
                    # labels shape [N, H, W]

                # Free up memory
                del data

                # Normalize inputs by subtracting the mean image, if it was given
                # (this is handled in dataset in torch datasets,
                # but must be done manually here for NVVL datasets)
                if mean_image_given:
                    if inputs_val.shape != mean_image.shape:
                        mean_image = mean_image[0, :, :].unsqueeze(0)
                        mean_image = mean_image.expand(inputs_val.shape[0], 3, -1, -1)
                        # mean_image will now be shape [N, C, H, W] for the appropriate N for this batch
                    inputs_val = inputs_val - mean_image

                # Convert these to cuda types if cuda is available
                if torch.cuda.is_available():
                    inputs_val, labels_val = inputs_val.cuda(), labels_val.cuda()
                
                # DEPRECATED - calling Variable should no longer be necessary, but leave in for now
                inputs_val = Variable(inputs_val)
                labels_val = Variable(labels_val)
                
                # train the model (forward propagation) on the inputs
                outputs_val = model(inputs_val)
                # Squeeze the outputs so it has shape [N, H, W] instead of [N, 1, H, W]
                outputs_val = outputs_val.squeeze(1)

                # Free up memory
                del inputs

                val_loss += self.loss_func(outputs_val, labels_val).item()
                
                # Free up memory
                del inputs_val, outputs_val, labels_val
            
            if nvvl_loader:
                val_loss /= len(val_loader_tuple[0])
            else:
                val_loss /= len(val_loader)
            
            self.val_loss_history.append(val_loss)
            # Check if this is the best validation loss so far. If so, save the current model state
            if val_loss < self.best_val_loss:
                if len(filename_args) < 3:
                    filename = 'trained_models/model_state_dict_best_loss_{:6f}.pth'.format(val_loss)
                else:
                    filename = 'trained_models/best_model_{}_lr2_batch{}_epoch{}.pth'.format(
                        filename_args['optim'],
                        filename_args['batchsize'],
                        filename_args['epoch_number'])
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': j + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': val_loss
                }, filename)
                tqdm.write("Checkpoint created with loss: {:6f}".format(val_loss))
            
            # Free up memory
            del val_loss
            
            # Print the average Train loss for the last epoch (avg of the logged losses, as decided by log_nth value)
            tqdm.write('[Epoch %i/%i] TRAIN NSS Loss: %f' % (j, num_epochs, sum(self.train_loss_history[-train_loss_logs:])/train_loss_logs))
            tqdm.write('[Epoch %i/%i] VAL NSS Loss: %f' % (j, num_epochs, self.val_loss_history[-1]))
            
        
        tqdm.write('FINISH.')
