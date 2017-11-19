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

import fan

def load_features(model, model_path):
    model_dict = model.state_dict()    
    pretrained_dict = model_zoo.load_url(model_path)

    updates = {k: v for k, v in model_dict.items() if not k.startswith('features')}
    updates.update({k: v for k, v in pretrained_dict.items() if k.startswith('features')})
    
    model.load_state_dict(updates)
    
    return model

class SqueezeNet(nn.Module):

    def __init__(self, num_classes=(10,4), extract=('4', '7', '12')):
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
        final_conv = [nn.Conv2d(512, self.num_classes[0], kernel_size=1),
                      nn.Conv2d(512, self.num_classes[1], kernel_size=1)]
        
        self.classifier_slide = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv[0],
            nn.ReLU(inplace=True),
            nn.AvgPool2d(256,ceil_mode=True)
        )
        self.classifier_domain = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv[1],
            nn.ReLU(inplace=True),
            nn.AvgPool2d(256,ceil_mode=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
        self._initialize()
                    
    def _initialize(self):
        
        m = load_features(self,'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth')
        self.features = m.features

    def forward(self, x, segment=False):
        out = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.extract_layers:
                out += [x]
        return self.classifier_slide(x).view(-1,self.num_classes[0]),\
               self.classifier_domain(x).view(-1,self.num_classes[1])