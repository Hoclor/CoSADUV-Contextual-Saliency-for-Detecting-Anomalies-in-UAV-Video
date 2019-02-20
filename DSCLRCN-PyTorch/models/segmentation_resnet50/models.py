import torch
import torch.nn as nn
import torchvision
from models.segmentation_resnet50 import resnet


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.001)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)

    def build_encoder(self, weights=''):
        pretrained = True if len(weights) == 0 else False
        orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
        net_encoder = Resnet(orig_resnet)

        # net_encoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            # Filter out unnecessary keys (fc.weight and fc.bias) - code from https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
            weight_dict = torch.load(weights, map_location=lambda storage, loc: storage)
            
            filtered_weights = {key: value for key, value in weight_dict.items() if 'fc.' not in key}
            net_encoder.load_state_dict(filtered_weights)
        return net_encoder

    def build_decoder(self, arch='c1_bilinear', fc_dim=512, num_class=150,
                      segSize=384, weights='', use_softmax=False):
        if arch == 'c1_bilinear':
            net_decoder = C1Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        elif arch == 'c5_bilinear':
            net_decoder = C5Bilinear(num_class=num_class,
                                     fc_dim=fc_dim,
                                     segSize=segSize,
                                     use_softmax=use_softmax)
        elif arch == 'psp_bilinear':
            net_decoder = PSPBilinear(num_class=num_class,
                                      fc_dim=fc_dim,
                                      segSize=segSize,
                                      use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(self.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage))
        return net_decoder

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1   = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.pool1 = orig_resnet.pool1
        
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# last conv, bilinear upsample
class C1Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(C1Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


# 2 conv with dilation=2, 2 conv with dilation=1, last conv, bilinear upsample
class C5Bilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False):
        super(C5Bilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        # convs, dilation=2
        self.conv1 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        self.conv2 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        # convs, dilation=1
        self.conv3 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(fc_dim, momentum=0.1)
        self.conv4 = nn.Conv2d(fc_dim, fc_dim, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(fc_dim, momentum=0.1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim, num_class, 1, 1, 0, bias=False)

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.conv_last(x)

        if not (x.size(2) == segSize[0] and x.size(3) == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x


# pyramid pooling, bilinear upsample
class PSPBilinear(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, segSize=384,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PSPBilinear, self).__init__()
        self.segSize = segSize
        self.use_softmax = use_softmax

        self.psp = []
        for scale in pool_scales:
            self.psp.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.psp = nn.ModuleList(self.psp)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, x, segSize=None):
        if segSize is None:
            segSize = (self.segSize, self.segSize)
        elif isinstance(segSize, int):
            segSize = (segSize, segSize)

        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                pool_scale(x),
                (input_size[2], input_size[3]),
                mode='bilinear'))
        psp_out = torch.cat(psp_out, 1)

        x = self.conv_last(psp_out)

        if not (input_size[2] == segSize[0] and input_size[3] == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')

        if self.use_softmax:
            x = nn.functional.softmax(x)
        else:
            x = nn.functional.log_softmax(x)
        return x
