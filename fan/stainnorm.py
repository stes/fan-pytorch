""" PyTorch implementation of a Stain Normalization network using FAN units
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

from . import transformer, layer

class StainNormalizerMultiFAN(nn.Module):
    """ Stain Normalization Network

    Network as presented in the original paper, but using both pre-trained
    VGG encoders and decoders.
    """

    def __init__(self, latent_size=32, domain_discrimination=False):

        super().__init__()

        self.domain_discrimination = domain_discrimination

        self.feature_extractor = transformer.get_encoder(depth=4)

        self.fan_s8 = layer.FeatureAwareNorm(128, 512, scale=4)
        self.fan_s4 = layer.FeatureAwareNorm(128, 256, scale=2)
        self.fan_s2 = layer.FeatureAwareNorm(128, 128, scale=1)

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
