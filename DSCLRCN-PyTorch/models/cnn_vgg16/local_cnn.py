"""LocalFeaturesCNN"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.places_vgg16 import PlacesCNN as PCNN


class LocalFeatsCNN(nn.Module):

    def __init__(self):
        super(LocalFeatsCNN, self).__init__()

        print('Loading weights for VGG16_CNN')
        vgg16_complete = torchvision.models.vgg16(pretrained=False)
        
        vgg16_weights = torch.load('models/cnn_vgg16/vgg16-397923af.pth')
        
        vgg16feats_weights = {}
        for k,v in vgg16_weights.items():
            if k[0]=='f':
                vgg16feats_weights[k[9:]] = v
        
        pretrained_dict = vgg16_complete.features.state_dict()
        pretrained_dict.update(vgg16feats_weights)
                
        conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=2)
        conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=2)
        conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=2)
        
        modified_convs = [nn.ReLU(inplace=True), conv5_1, nn.ReLU(inplace=True), conv5_2, nn.ReLU(inplace=True), conv5_3, nn.ReLU(inplace=True)]
        
        self.feats = nn.Sequential(*list(vgg16_complete.features)[0:23], *modified_convs)

        model_dict = self.feats.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        self.feats.load_state_dict(pretrained_dict)
        
        #self.conv6_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=4)
        #self.conv6_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=4)
        #self.conv7_1 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=4)
        #self.conv7_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=4)
        
        self.deconv = nn.ConvTranspose2d(512, 128, 7)
                        

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        
        x = self.feats(x)
        
        #print(x.size())
        #x = F.relu(self.conv6_1(x))
        #print(x.size())
        #x = F.relu(self.conv6_2(x))
        
        #x = F.relu(self.conv7_1(x))
        
        #x = F.relu(self.conv7_2(x))
        
        x = self.deconv(x)
        
        xVector = x.view(x.size(0), x.size(1), -1, 1)
        norm = xVector.norm(p=2, dim=2, keepdim=True)
        
        x_norm = x.div(norm.expand_as(x) + 1e-8)*100

        return x_norm

    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
