import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import numpy as np
from pyroResNet import resnet50


__all__ = ['SqueezeNet', 'squeezenet11_pretrained', 'DenseAEC']


model_urls = {
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


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
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000, extract=('2', '5', '12')):
        super(SqueezeNet, self).__init__()
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
            nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        # print(self.features._modules.items())

    def forward(self, x):
        out = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.extract_layers:
                out += [x]
        return out


def squeezenet11_pretrained(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(**kwargs)
    model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


class FeatureAwareNorm(nn.Module):

    def __init__(self, in_x, in_z, scale):
        
        super(FeatureAwareNorm, self).__init__()
        self.mul_gate = nn.Conv2d(in_z, in_x, kernel_size=1)
        self.add_gate = nn.Conv2d(in_z, in_x, kernel_size=1)
        self.mul_upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.add_upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, np.sqrt(6/(m.in_channels+m.out_channels)))
                m.bias.data.fill_(0)

    def forward(self, x, z):
        # print(x.size(), z.size())
        gamma = self.mul_upsample(self.sigm(self.mul_gate(z)))
        beta = self.add_upsample(self.relu(self.add_gate(z)))

        return torch.mul(x, gamma) + beta


class TissueNormalizer(nn.Module):

    def __init__(self, latent_size=8):
    
        super(TissueNormalizer, self).__init__()
        self.feature_extractor = resnet50(pretrained=True)
#         self.feature_extractor = squeezenet11_pretrained()
#        for param in self.feature_extractor.parameters():
#            param.requires_grad = False

        self.transformer = nn.Conv2d(3, latent_size, kernel_size=1)

        self.fan_s8 = FeatureAwareNorm(latent_size, 1024, scale=16)
        self.fan_s4 = FeatureAwareNorm(latent_size, 512, scale=8)
        self.fan_s2 = FeatureAwareNorm(latent_size, 64, scale=2)

        self.inv_transformer = nn.Conv2d(latent_size, 3, kernel_size=1)  # not sure weight tying is possible here

        # init remaining parameters
        self.transformer.weight.data.normal_(0, np.sqrt(6/(3+latent_size)))
        self.inv_transformer.weight.data.normal_(0, np.sqrt(6/(3+latent_size)))

    def get_training_params(self):
        return list(self.transformer.parameters()) + \
                list(self.fan_s8.parameters()) + \
                list(self.fan_s4.parameters()) + \
                list(self.fan_s2.parameters()) + \
                list(self.inv_transformer.parameters())

    def forward(self, x):
        z0, z1, z2, z3 = self.feature_extractor(x)
        # print(z0.size(), z2.size(), z3.size())
        y = self.transformer(x)
        y = self.fan_s8(y, z3)
        y = self.fan_s4(y, z2)
        y = self.fan_s2(y, z0)
        y = self.inv_transformer(y)
        return y, z3


class TissueClassifier(nn.Module):
    def __init__(self):
        super(TissueClassifier, self).__init__()
        # TODO: define classifier modules
        # TODO: init components
        pass

    def forward(self, x, z):
        # TODO: connect classifier
        pass


class DenseAEC(nn.Module):

    def __init__(self, input_size, l_size=32, p_dropout=0.5, output_size=10):
        super(DenseAEC, self).__init__()
        self.fc1 = nn.Linear(input_size, l_size)
        self.dropout = nn.Dropout(p=p_dropout, inplace=False)
        self.fc2 = nn.Linear(l_size, input_size)
        self.sigm = nn.Sigmoid()
        self.elu = nn.ELU()
        self.clf_output = output_size
        if self.clf_output > 0:
            self.fcc = nn.Linear(l_size, self.clf_output)
            self.smax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(6/(m.in_features+m.out_features)))
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        y = self.sigm(self.fc2(x))

        if self.clf_output == 0:
            return y
        else:
            z = self.smax(self.fcc(x))
            return y, z


class ConvolutionalAEC_VGG(nn.Module):

    def __init__(self):
        super(ConvolutionalAEC_VGG, self).__init__()

    def forward():
        pass
