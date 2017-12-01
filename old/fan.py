""" PyTorch implementation of Feature Aware Normalization

This module contains pytorch modules. To directly apply this model,
have a look at the ``stainnorm.py`` module
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import functional, init

### Helper functions

def update_with_pretrained_weights(model, model_path):
    model_dict = model.state_dict()    
    pretrained_dict = model_zoo.load_url(model_path)

    diff = {k: v for k, v in model_dict.items() if \
            k in pretrained_dict and pretrained_dict[k].size() != v.size()}
    
    pretrained_dict.update(diff)
    model.load_state_dict(pretrained_dict)
    
    return model, diff

def load_features(model, model_path):
    model_dict = model.state_dict()    
    pretrained_dict = model_zoo.load_url(model_path)

    updates = {k: v for k, v in model_dict.items() if not k.startswith('features')}
    updates.update({k: v for k, v in pretrained_dict.items() if k.startswith('features')})
    
    model.load_state_dict(updates)
    
    return model


def upsample(upscale_factor, inp_channel, outp_channel, mode='bilinear'):
    
    if mode == 'subpixel':
        return nn.Sequential(
            nn.Conv2d(inp_channel, outp_channel * upscale_factor ** 2, (1, 1), (1, 1), (0, 0)),
            nn.PixelShuffle(upscale_factor)
            )
    else:
        return nn.Sequential(
            torch.nn.ReflectionPad2d(2),
            
            torch.nn.Dropout(p=0.25),
            Fire(inp_channel, inp_channel//2, inp_channel//2, inp_channel//2),
            #nn.InstanceNorm2d(inp_channel, affine=True, momentum=0.),
            nn.SELU(inplace=True),
            
            torch.nn.Dropout(p=0.25),
            Fire(inp_channel, inp_channel//2, outp_channel//2, outp_channel//2),
            #nn.InstanceNorm2d(outp_channel, affine=True, momentum=0.),
            #nn.SELU(inplace=True),
            
            nn.Upsample(scale_factor=upscale_factor, mode=mode)
            )

def make_conv_layers(in_channels, out_channels, kernel_size=1,\
                     batch_norm=False, relu=True):
    layers = []
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
    if batch_norm:
        layers += [conv2d, nn.InstanceNorm2d(out_channels, affine=True,\
                                             momentum=0.)]
    else:
        layers += [conv2d]
        
    if relu:
        layers += [nn.SELU(inplace=True)]
    return nn.Sequential(*layers)

### Network Modules

class SpatialNormalization(nn.Module):
    
    def __init__(self):
        
        pool     = nn.AvgPool2d()
        
    def forward(self, x):
        
        mu    = upsample(pool(x))
        sigma = upsample(pool((x - upsample(mu))**2))
        
        return nn.mul(x - mu, 1/sigma)
        

class Transformer(nn.Module):
    """ Stain Normalization Network using Feature Aware Normalization
    """

    def __init__(self, inp_channel, mid_channel, out_channel, **kwargs):

        super(Transformer, self).__init__()

        self.conv1 = make_conv_layers(inp_channel, mid_channel, **kwargs)
        self.conv2 = make_conv_layers(mid_channel, out_channel, **kwargs)
        self.conv3 = make_conv_layers(out_channel, out_channel, **kwargs)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)

        return y
 
    
class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_x, in_z, scale):

        super().__init__()

        # layers
        self.mul_gate = upsample(upscale_factor=1, inp_channel=in_z,\
                                 outp_channel=in_x)
        self.add_gate = upsample(upscale_factor=1, inp_channel=in_z,\
                                 outp_channel=in_x)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.pool = nn.AvgPool2d(1000, ceil_mode=True)
        
        self.norm = nn.InstanceNorm2d(in_x, momentum=0., affine=False)

        # parameters
        self.inp_channel = in_x

    def forward(self, x, z):
        gamma = self.sigm(self.pool(self.mul_gate(z))) * 3.
        beta  = self.pool(self.add_gate(z))
        x = self.norm(x)

        return torch.mul(x, gamma) + beta 

class FeatureAwareNorm(nn.Module):

    def __init__(self, in_x, in_z, scale):

        super().__init__()

        # layers
        self.mul_gate = upsample(upscale_factor=scale, inp_channel=in_z,\
                                 outp_channel=in_x)
        self.add_gate = upsample(upscale_factor=scale, inp_channel=in_z,\
                                 outp_channel=in_x)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.norm = nn.InstanceNorm2d(in_x, momentum=0., affine=False)

        # parameters
        self.inp_channel = in_x

    def forward(self, x, z):
        gamma = self.tanh(self.mul_gate(z))
        beta  = self.add_gate(z)
        
        x = self.norm(x)

        return torch.mul(x, gamma) + beta


## our feature extractor, can be extended

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=0)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x[:,:,1:-1,1:-1])),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000, extract=('4', '7', '12')):
        super().__init__()
        self.extract_layers = extract
        self.num_classes = num_classes
        self.features = nn.Sequential(
            #torch.nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(inplace=True),     
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                torch.nn.ReflectionPad2d(2)),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                torch.nn.ReflectionPad2d(2)),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                torch.nn.ReflectionPad2d(4)),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        
        if self.num_classes is not None:
            # Final convolution is initialized differently form the rest
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AvgPool2d(256,ceil_mode=True)
            self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is final_conv:
                        init.normal(m.weight.data, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

        self._initialize()
                    
    def _initialize(self):
        
        if self.num_classes is None:
            m = load_features(self, 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth')
        else:
            m, diff = update_with_pretrained_weights(self,
                                         'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth')
        self.features = m.features

    def forward(self, x, segment=False):
        out = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.extract_layers:
                out += [x]
        if self.num_classes is None:
            return out
        y = self.classifier(x)
        if segment:
            y = self.upsample(y)
        else:
            y = self.pool(y)
            y = y.view(-1, self.num_classes)
        return out, y