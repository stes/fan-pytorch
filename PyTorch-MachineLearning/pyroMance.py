import numpy as np
import torch.nn as nn
import torch

def batch_iterator(dataset_size, batch_size, permute=True):
    """
    Iterates over permutated indices for the given dataset_size and batch_size
    @param dataset_size: size of the dataset, e.g. X_train.shape[0]
    @param batch_size: size of the batches, e.g. 64
    """
    idc = np.arange(dataset_size)
    if permute:
        idc = np.random.permutation(idc)
    for i in range(0, dataset_size, batch_size):
        yield idc[i:min(i+batch_size, dataset_size)]


def skrew_labels(y_labels, fraction=0.5):
    """
    Ruins a certain fraction of your labels, by setting them to the target label.
    """
    idc = np.random.permutation(np.arange(y_labels.shape[0]))
    upper = np.int32(fraction*y_labels.shape[0])
    new_labels = np.copy(y_labels)
    new_labels[idc[:upper], :] = 0
    return new_labels


class SelectiveCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SelectiveCrossEntropyLoss, self).__init__()
        self.min = 1e-6
        self.max = 1-self.min

    def forward(self, prediction, target):
        t = target.float()
        p = torch.clamp(prediction, min=self.min, max=self.max)
        weights = torch.clamp(torch.sum(t, 1), min=.0, max=1.)
        cce_pos = torch.sum(t*torch.log(p), 1)
        cce_neg = torch.sum((1-t)*torch.log(1.-p), 1)
        cce = - (cce_pos + cce_neg)
        return torch.sum(cce*weights ) / torch.clamp(torch.sum(weights ), min=1.)


class FeatureAwareNorm(nn.Module):
    def __init__(self, in_x, in_z, scale):
        self.mul_gate = nn.Conv2D(in_x, in_z, kernel_size=1)
        self.add_gate = nn.Conv2D(in_x, in_z, kernel_size=1)
        self.mul_upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.add_upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.mul_activation = torch.sigmoid
        self.add_activation = torch.clamp

    def forward(self, x, z):
        gamma = self.mul_upsample(self.mul_activation(self.mul_gate(z)))
        beta = self.add_upsample(self.add_activation(self.add_gate(z)))

        return torch.mul(x, gamma) + beta
