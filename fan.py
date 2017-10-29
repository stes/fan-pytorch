""" PyTorch implementation of Feature Aware Normalization
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import numpy as np

from torchvision.models.squeezenet import Fire
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import functional, init

### Helper functions

def upsample(upscale_factor, inp_channel, outp_channel):
    return nn.Sequential(
        nn.Conv2d(inp_channel, outp_channel * upscale_factor ** 2, (3, 3), (1, 1), (1, 1)),
        nn.PixelShuffle(upscale_factor)
        )

def make_conv_layers(in_channels, out_channels, kernel_size=1, batch_norm=False):
    layers = []
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
    if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

### Network Modules

class Transformer(nn.Module):
    """ Stain Normalization Network using Feature Aware Normalization
    """

    def __init__(self, inp_channel, mid_channel, out_channel):

        super(Transformer, self).__init__()

        self.conv1 = make_conv_layers(inp_channel, mid_channel)
        self.conv2 = make_conv_layers(mid_channel, out_channel)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        return y

class StainNormalizer(nn.Module):
    """ Stain Normalization Network

    features:
        incremental list of features to use as the inputs to FAN modules
    """

    def __init__(self, latent_size=32):

        super().__init__()

        self.feature_extractor = SqueezeNet(num_classes=10)

        self.fan_s8 = FeatureAwareNorm(latent_size, 512, scale=8)
        self.fan_s4 = FeatureAwareNorm(latent_size, 256, scale=4)
        self.fan_s2 = FeatureAwareNorm(latent_size, 128, scale=2)

        self.transformer     = Transformer(3,16,32)
        self.inv_transformer = Transformer(32,16,3)

    def forward(self, x):
        (z2, z4, z8), d = self.feature_extractor(x)

        y = self.transformer(x)
        y = self.fan_s8(y, z8)
        y = self.fan_s4(y, z4)
        y = self.fan_s2(y, z2)
        y = self.inv_transformer(y)

        return y, d

class FeatureAwareNorm(nn.Module):

    def __init__(self, in_x, in_z, scale, eps=1e-5):

        super().__init__()

        # layers
        self.mul_gate = upsample(upscale_factor=scale, inp_channel=in_z, outp_channel=in_x)
        self.add_gate = upsample(upscale_factor=scale, inp_channel=in_z, outp_channel=in_x)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

        # parameters
        self.eps = eps
        self.inp_channel = in_x

    def forward(self, x, z):
        gamma = self.sigm(self.mul_gate(z))
        beta  = self.relu(self.add_gate(z))

        mean = torch.zeros(self.inp_channel)
        var  = torch.ones(self.inp_channel)

        x = functional.batch_norm(
                    input=x, running_mean=mean, running_var=var,
                    weight=None, bias=None, training=True,
                    momentum=1., eps=self.eps)

        return torch.mul(x, gamma) + beta


## our feature extractor, can be extended

class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000, extract=('4', '7', '12')):
        super().__init__()
        self.extract_layers = extract
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(256,ceil_mode=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.extract_layers:
                out += [x]
        y = self.classifier(x)
        y = y.view(-1, self.num_classes)
        return out, y
