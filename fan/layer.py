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

#### Helper functions & layers ####

def upsample(upscale_factor, inp_channel, outp_channel, mode='bilinear'):
    """ Helper function for creating upsampling layers
    
    Builds an upsampling module for converting an input of size
    `(N, inp_channel, W, H)` into `(N, outp_channel, W*upscale_factor, H*upscale_factor)`.
    The module can either make use of subpixel convolutions 
    
    upscale_factor:
        Upsampling factor
    inp_channel:
        Number of input channels
    outp_channel:
        Number of output channels
    mode:
        upsampling mode passed on to `torch.nn.Upsample` or `subpixel` for
        subpixel convolution
        
    """
    
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
            nn.SELU(inplace=True),
            
            torch.nn.Dropout(p=0.25),
            Fire(inp_channel, inp_channel//2, outp_channel//2, outp_channel//2),
            nn.Upsample(scale_factor=upscale_factor, mode=mode)
            )
    
class Fire(nn.Module):
    """ Fire module, taken from the PyTorch SqueezeNet implementation """

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
    
    
#### Normalization Layers ####

class AdaptiveInstanceNorm(nn.Module):
    """ Adaptive Instance Normalization
    """

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
    """ Feature Aware Normalization
    """

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