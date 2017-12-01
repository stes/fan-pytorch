import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import functional, init

import transformer, fan

class StainNormalizerSingleFAN(nn.Module):
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
        
        
class StainNormalizerMultiFAN(nn.Module):
    """ Stain Normalization Network
    
    Network as presented in the original paper, but using both pre-trained
    VGG encoders and decoders.

    features:
        incremental list of features to use as the inputs to FAN modules
    """

    def __init__(self, latent_size=32, domain_discrimination=False):

        super().__init__()
        
        self.domain_discrimination = domain_discrimination

        self.feature_extractor = transformer.get_encoder(depth=4)

        self.fan_s8 = fan.FeatureAwareNorm(128, 512, scale=4)
        self.fan_s4 = fan.FeatureAwareNorm(128, 256, scale=2)
        self.fan_s2 = fan.FeatureAwareNorm(128, 128, scale=1)

        self.transformer     = transformer.get_encoder(depth=2)
        self.inv_transformer = transformer.get_decoder(depth=2)
        
        self._initialize()
        
    def _initialize(self):

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.inv_transformer.parameters():
            param.requires_grad = False
    
    def get_training_params(self, finetune=False):
        trainable_modules = [
                             self.fan_s8.parameters(),
                             self.fan_s4.parameters(),
                             self.fan_s2.parameters()]
        
        for mod in trainable_modules:
            for param in mod:
                yield param

    def forward(self, x, y=None):
        z2, z4, z8 = self.feature_extractor(x)
        
        transf = y = self.transformer(x)
        y = self.fan_s8(y, z8)
        y = self.fan_s4(y, z4)
        y = self.fan_s2(y, z2)
        y = self.inv_transformer(y + transf)

        return y