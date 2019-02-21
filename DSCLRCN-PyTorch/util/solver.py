from random import shuffle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from random import *

from tqdm import tqdm

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    default_SGD_args = {"lr": 1e-2,
                        "weight_decay": 0.0005,
                        "momentum": 0.9}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.KLDivLoss(), location='ncc'):
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

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, filename_args={}):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        # Move the model to cuda first, if applicable, so optimiser is initialized properly
        if torch.cuda.is_available():
            model.cuda()
        
        # Add the parameters to the optimiser as two groups: the pretrained parameters (PlacesCNN, ResNet50), and the other parameters
        # This allows us to set the lr of the two groups separately (other params as the lr given as input, pretrained params as this lr * 0.1)
        pretrained_parameters = [param for name,param in model.named_parameters() if name.startswith('local_feats') or name.startswith('context')]
        other_parameters = [param for name,param in model.named_parameters() if not (name.startswith('local_feats') or name.startswith('context'))]
        pretrained_param_group = self.optim_args.copy()
        pretrained_param_group['lr'] *= 1e-1
        pretrained_param_group['params'] = pretrained_parameters
        
        optim = self.optim(other_parameters, **self.optim_args)
        optim.add_param_group(pretrained_param_group)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        
        # Create the scheduler to allow lr adjustment
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=1/2.5)

        tqdm.write('START TRAIN.')
        
        nIterations = num_epochs*iter_per_epoch
        
        epoch_loop = range(num_epochs)
        if self.location != 'ncc':
            epoch_loop = tqdm(epoch_loop)

        # Epoch
        for j in epoch_loop:
            train_loss_logs = 0
            # Downscale the learning rate by a factor of 2.5 (i.e. multiply by 1/2.5) every epoch
            scheduler.step()
            print('Starting an epoch')

            # Set the model to training mode
            model.train()

            if self.location != 'ncc':
                train_loop = enumerate(tqdm(train_loader), 0)
            else:
                print('enumerating train_loader')
                train_loop = enumerate(train_loader, 0)
                print('train_loader enumerated')

            # Batch of items in training set
            for i, data in train_loop:
                
                it = j*iter_per_epoch + i
                # Load the items in this batch and their labels from the train_loader
                inputs, labels = data
                # Unsqueeze labels so they're shaped as [10, 96, 128, 1]
                labels = labels.unsqueeze(3)

                # Convert these to cuda types if cuda is available
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                # DEPRECATED - calling Variable should no longer be necessary, but leave in for now
                inputs = Variable(inputs)
                labels = Variable(labels)
                
                # train the model (forward propgataion) on the inputs
                outputs = model(inputs)
                # transpose the outputs so it's in the order [N, H, W, C] instead of [N, C, H, W]
                outputs = outputs.transpose(1, 3)
                outputs = outputs.transpose(1, 2)
                
                loss = self.loss_func(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if it%log_nth==0:
                    tqdm.write('[Iteration %i/%i] TRAIN loss: %f' % (it, nIterations, loss))
                    self.train_loss_history.append(loss.item())
                    train_loss_logs += 1
                
                # Free up memory
                del loss, outputs
            
            model.eval()
            
            rand_select = randint(0, len(val_loader)-1)
            for ii, data in enumerate(val_loader, 0):
                if rand_select == ii:
                    inputs, labels = data
                    # Unsqueeze labels so they're shaped as [batch_size, H, W, 1]
                    labels = labels.unsqueeze(3)
                    
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs_val = Variable(inputs)
                    labels_val = Variable(labels)
                    
                    outputs_val = model(inputs_val)
                    # transpose the outputs so it's in the order [N, H, W, C] instead of [N, C, H, W]
                    outputs_val = outputs_val.transpose(1, 3)
                    outputs_val = outputs_val.transpose(1, 2)
                    
                    # Normalize the labels into range [0, 1] with sum of values in each image = 1, as this is how the output is structured
                    labels_sum = torch.sum(labels.contiguous().view(labels.size(0),-1), dim=1)
                    labels /= labels_sum.contiguous().view(*labels_sum.size(), 1, 1, 1).expand_as(labels)
                    
#                     outputs_val = torch.log(outputs_val)
                    
                    val_loss = self.loss_func(outputs_val, labels_val)
                    self.val_loss_history.append(val_loss.item())
                    # Check if this is the best validation loss so far. If so, save the current model state
                    if val_loss.item() < self.best_val_loss:
                        if len(filename_args) < 4:
                            filename = 'trained_models/model_state_dict_best_loss_{:6f}.pth'.format(val_loss.item())
                        else:
                            filename = 'trained_models/best_model_{}_{}_lr2_batch{}_epoch{}.pth'.format(
                                filename_args['net_type'],
                                filename_args['optim'],
                                filename_args['batchsize'],
                                filename_args['epoch_number'])
                        self.best_val_loss = val_loss.item()
                        torch.save({
                            'epoch': j + 1,
                            'state_dict': model.state_dict(),
                            'best_accuracy': val_loss.item()
                        }, filename)
                        tqdm.write("Checkpoint created with loss: {:6f}".format(val_loss.item()))
                    
                    # Free up memory
                    del val_loss, outputs_val
                    
            # Print the average Train loss for the last epoch (avg of the logged losses, as decided by lo_nth value)
            tqdm.write('[Epoch %i/%i] TRAIN NSS Loss: %f' % (j, num_epochs, sum(self.train_loss_history[-train_loss_logs:])/train_loss_logs))
            tqdm.write('[Epoch %i/%i] VAL NSS Loss: %f' % (j, num_epochs, self.val_loss_history[-1]))
            
        
        tqdm.write('FINISH.')
