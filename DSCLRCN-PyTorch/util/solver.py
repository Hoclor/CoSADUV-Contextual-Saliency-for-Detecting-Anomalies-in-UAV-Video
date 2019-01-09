from random import shuffle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from random import *

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}
    default_SGD_args = {"lr": 1e-2,
                        "weight_decay": 0.0005,
                        "momentum": 0.9}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.KLDivLoss()):
        if optim == torch.optim.Adam:
            optim_args_merged = self.default_adam_args.copy()
        else:
            optim_args_merged = self.default_SGD_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

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



        print('START TRAIN.')
        
        nIterations = num_epochs*iter_per_epoch
        
        # Epoch
        for j in range(num_epochs):
            # Batch of items in training set
            for i, data in enumerate(train_loader, 0):
                
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
                # Set the model to training mode
                model.train()
                # train the model (forward propgataion) on the inputs
                outputs = model(inputs)
                # transpose the outputs so it's in the order [N, H, W, C] instead of [N, C, H, W]
                outputs = outputs.transpose(1, 3)
                outputs = outputs.transpose(1, 2)
        
                # Apply a natural logarithm to the outputs, i.e. outputs = log_e(outputs)
                outputs = torch.log(outputs)
                # Normalize the labels by dividing each value by the sum of values of that item
                # Create a list of label sums (i.e. one entry per item, each entry is the sum of values in that label)
                labels_sum = torch.sum(labels.contiguous().view(labels.size(0),-1), dim=1)
                
                labels /= labels_sum.contiguous().view(*labels_sum.size(), 1, 1, 1).expand_as(labels)
                
                loss = self.loss_func(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if it%log_nth==0:
                    print('[Iteration %i/%i] TRAIN loss: %f' % (it, nIterations, loss))
                    self.train_loss_history.append(loss.item())
            
            model.eval()
            
            rand_select = randint(0, len(val_loader)-1)
            for ii, data in enumerate(val_loader, 0):
                inputs, labels = data
                # Unsqueeze labels so they're shaped as [10, 96, 128, 1]
                labels = labels.unsqueeze(3)
                if rand_select == ii:
                    if torch.cuda.is_available():
                        inputs, labels = inputs.cuda(), labels.cuda()
                    inputs_val = Variable(inputs)
                    labels_val = Variable(labels)
                    
                    outputs_val = model(inputs_val)
                    # transpose the outputs so it's in the order [N, H, W, C] instead of [N, C, H, W]
                    outputs_val = outputs_val.transpose(1, 3)
                    outputs_val = outputs_val.transpose(1, 2)
                    outputs_val = torch.log(outputs_val)
                    
                    labels_sum = torch.sum(labels.contiguous().view(labels.size(0),-1), dim=1)
                    labels /= labels_sum.contiguous().view(*labels_sum.size(), 1, 1, 1).expand_as(labels)
                    val_loss = self.loss_func(outputs_val, labels_val)
                    self.val_loss_history.append(val_loss.item())
                    # Check if this is the best validation loss so far. If so, save the current model state
                    if val_loss.item() < self.best_val_loss:
                        if len(filename_args) < 3:
                            filename = 'pretrained/model_state_dict_best_loss_{:6f}.pth'.format(val_loss.item())
                        else:
                            filename = 'pretrained/best_model_{}_100_lr4_batch{}_epoch{}.pth'.format(
                                filename_args['net_type'],
                                filename_args['batchsize'],
                                filename_args['epoch_number'])
                        self.best_val_loss = val_loss.item()
                        torch.save({
                            'epoch': j + 1,
                            'state_dict': model.state_dict(),
                            'best_accuracy': val_loss.item()
                        }, filename)
                    
            print('[Epoch %i/%i] TRAIN KLD Loss: %f' % (j, num_epochs, loss.item()))
            print('[Epoch %i/%i] VAL KLD Loss: %f' % (j, num_epochs, val_loss.item()))
            
        
        print('FINISH.')
