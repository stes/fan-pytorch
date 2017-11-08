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

class StainNormalizer2(nn.Module):
    """ Stain Normalization Network

    features:
        incremental list of features to use as the inputs to FAN modules
    """

    def __init__(self, latent_size=32):

        super().__init__()

        self.feature_extractor = SqueezeNet(None, extract=('4', '7', '12'))
        
        self.transformer     = Transformer(3,16,32,relu=True,batch_norm=True)
        self.inv_transformer = Transformer(32,16,3,relu=True,batch_norm=False)

        self.adain_s8 = AdaptiveInstanceNorm(latent_size, 512, scale=8)
        self.fan_s8   = FeatureAwareNorm(latent_size, 512, scale=8)
        #self.fan_s4 = FeatureAwareNorm(latent_size, 256, scale=4)
        #self.fan_s2 = FeatureAwareNorm(latent_size, 128, scale=2)
        
        
        self.norm = nn.InstanceNorm2d(latent_size, momentum=0., affine=False)
        
        self._initialize()
        
    def _initialize(self):

        for param in self.feature_extractor.features.parameters():
            param.requires_grad = False
    
    def get_training_params(self, finetune=False):
        trainable_modules = [self.adain_s8.parameters(),
                             self.fan_s8.parameters()]
        
        if not finetune:
            trainable_modules += [self.transformer.parameters(),
                                 self.inv_transformer.parameters()]
        
        for mod in trainable_modules:
            for param in mod:
                yield param
    
    def norm_input(self, x):
        return x / 255.
    
    def forward(self, content, style):
        
        train = True
        
        content = self.norm_input(content)
        style   = self.norm_input(style)
        
        y_content = self.transformer(content)
        y_style   = self.transformer(style)
        z_style   = self.feature_extractor(style)
        
        N, C, W, H = y_style.size()
        gamma = y_style.view(N,C,W*H).std(dim=2).view(N,C,1,1)
        beta  = y_style.view(N,C,W*H).mean(dim=2).view(N,C,1,1)
        
        y_normed = self.norm(y_content)
        y_normed = torch.mul(y_normed, gamma) + beta
        
        #y_normed = self.adain_s8(y_normed,  z_style[2])
        #y_normed = self.fan_s8(y_normed,  z_style[2])
        #y_normed = self.fan_s4(y_normed,  z_style[1])
        #y_normed = self.fan_s2(y_normed,  z_style[0])
        
        normed = self.inv_transformer(y_normed)
        normed = 0.9 * torch.clamp(normed, 0, 1.5) + 0.1 * normed
        
        if train:
            # Additional computation for training phase
            z_content    = self.feature_extractor(content)
            z_normed     = self.feature_extractor(normed)
            
            return normed, z_content, z_style, z_normed

        return normed
        
        
class StainNormalizer(nn.Module):
    """ Stain Normalization Network

    features:
        incremental list of features to use as the inputs to FAN modules
    """

    def __init__(self, latent_size=32, domain_discrimination=False):

        super().__init__()
        
        self.domain_discrimination = domain_discrimination

        self.feature_extractor = torch.load('classifier_split0.pth') #SqueezeNet(num_classes=8)

        self.fan_s8 = FeatureAwareNorm(latent_size, 512, scale=8)
        self.fan_s4 = FeatureAwareNorm(latent_size, 256, scale=4)
        self.fan_s2 = FeatureAwareNorm(latent_size, 128, scale=2)

        self.transformer     = Transformer(3,16,32,relu=True,batch_norm=True)
        self.inv_transformer = Transformer(32,16,3,relu=False,batch_norm=True)
        
        self._initialize()
        
    def _initialize(self):

        if not self.domain_discrimination:
            for param in self.feature_extractor.features.parameters():
                param.requires_grad = False
    
    def get_training_params(self, finetune=False):
        trainable_modules = [self.feature_extractor.classifier.parameters(),
                             self.fan_s8.parameters(),
                             self.fan_s4.parameters(),
                             self.fan_s2.parameters()]
        
        if not finetune:
            trainable_modules += [self.transformer.parameters(),
                                 self.inv_transformer.parameters()]
        
        if self.domain_discrimination:
            trainable_modules += [self.feature_extractor.features.parameters()]
        
        for mod in trainable_modules:
            for param in mod:
                yield param

    def forward(self, x):
        (z2, z4, z8), d = self.feature_extractor(x/128. - 1)

        y = self.transformer(x)
        y = self.fan_s8(y, z8)
        y = self.fan_s4(y, z4)
        y = self.fan_s2(y, z2)
        y = self.inv_transformer(y)
        
        if self.domain_discrimination:
            _, dnorm = self.feature_extractor(x)
            return y, d, dnorm

        return y, d
 
    
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
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.norm = nn.InstanceNorm2d(in_x, momentum=0., affine=False)

        # parameters
        self.inp_channel = in_x

    def forward(self, x, z):
        gamma = self.sigm(self.mul_gate(z)) * 3.
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