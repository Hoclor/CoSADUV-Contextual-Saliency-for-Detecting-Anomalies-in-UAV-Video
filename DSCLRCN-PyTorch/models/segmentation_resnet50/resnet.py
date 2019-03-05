import os
import sys
import torch
import torch.nn as nn
import math

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


__all__ = ['resnet50']

class ResNet50(nn.Module):
    def __init__(self, num_classes=365):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 2
        self.layer1 = self._make_layer(ResidualBlock, 64, 3)
        
        # Block 3
        self.layer2 = self._make_layer(ResidualBlock, 128, 4, stride=2)
        
        # Block 4
        self.layer3 = self._make_layer(ResidualBlock, 256, 6, dilation=2)
        
        # Block 5
        self.layer4 = self._make_layer(ResidualBlock, 512, 3, dilation=4)
        
        # FC layer used for training
        self.fc = nn.Linear(512 * ResidualBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        # We need to downsample the input for the residual step if stride is not 1, or if it is not the same size as the output of the last layer (planes * block.expansion), as the residual step is after this layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        # Update self.inplanes to planes * block.expansion, as each block expands the channels of the tensor
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=2, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # padding = dilation to ensure width and height remain the same
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet50(**kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']))
    return model

def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
