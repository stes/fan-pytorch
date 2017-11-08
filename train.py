import numpy as np

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

import sys
from torch.autograd import Variable
import time

import fan

class StainNormalizer():
    
    """ Wrapper around style normalization networks
    
    Access as a regular ``scikit-learn'' model using
    
    ``` python
    >>> X = get_data()
    >>> model = StainNormalizer()
    >>> model.fit(X)
    >>> X_ = model.transform(X)
    ```
    
    """
    
    def __init__(self, learning_rate = 1e-3, weights = (1.,1.,1.),\
                 use_gpu=True, device_id=None, verbose=True):
        
        self.learning_rate = learning_rate
        self.weights = weights
        self.verbose = verbose      

        self.model     = fan.StainNormalizer2()
        self.optimizer = torch.optim.Adam(self.model.get_training_params(), lr = self.learning_rate)

        if use_gpu: self.cuda(device_id)
        else: self.cpu()

        

    def __repr__(self):
        return repr(self.model)
    
    def __apply__(self, iterator):
        return self.transform(iterator)
    
    def cpu(self):
        self.use_gpu = False
        self.device_id = None
    
    def cuda(self, device_id):
        self.use_gpu   = True
        self.device_id = device_id
        self.model.cuda(device_id)
    
    def loss(self, z_content, z_style, z_normed):

        def get_mse(n=1):
            mse = []
            for i in range(n):
                mse.append(torch.nn.MSELoss())
                if self.use_gpu:
                    mse[-1].cuda(self.device_id)
            return mse

        N, C = z_content[0].size()[:2]

        idc_content = [len(z_content)-1] #range(len(z_content))
        idc_style = range(len(z_style))

        loss_content = []
        loss_style = []

        for i in idc_content:
            mse, = get_mse(1)        
            loss_content += [self.weights[0] * mse(z_normed[i],
                                 z_content[i])]

        for i in idc_style:
            mse_mean, mse_sigma = get_mse(2)

            mean_style   = z_style[i].view(N,C,-1).mean(2)
            mean_normed  = z_normed[i].view(N,C,-1).mean(2)

            sigma_style   = z_style[i].view(N,C,-1).std(2)
            sigma_normed  = z_normed[i].view(N,C,-1).std(2)

            loss_style += [self.weights[1] * mse_mean(mean_normed,mean_style),
                           self.weights[2] * mse_sigma(sigma_normed,sigma_style)]


        return sum(loss_content), sum(loss_style)

    def _update(self, inputs, targets):
        """ Runs an update on a single sample """

        def to_torchvar(x):
            x = Variable(torch.from_numpy(x).float())
            if self.use_gpu:
                return x.cuda(self.device_id)
            return x

        inputs  = to_torchvar(inputs)
        targets = to_torchvar(targets)

        self.optimizer.zero_grad()

        normed, z_content, z_style, z_normed = self.model(inputs, targets)
        losses = self.loss(z_content, z_style, z_normed)

        sum(losses).backward()
        self.optimizer.step()

        return [l.cpu().data[0] for l in losses]

    def fit(self, iterator):
        # Training to one domain
        losses = []

        self.model.train()
        tic = time.time()
        for data in iterator:
            inputs, targets  = data
            losses.append(self._update(inputs, targets))
            toc = time.time()
            if self.verbose and (toc - tic > 5) and (len(losses) > 20):
                print('Content {:.3f} Style {:.3f}'.format(*np.array(losses)[-20:].mean(axis=0)))
                tic = time.time()
        return np.array(losses)


    def transform(self, iterator):
        self.model.eval()

        def to_torchvar(x):
            x = Variable(torch.from_numpy(x).float())
            if self.use_gpu:
                return x.cuda(self.device_id)
            return x

        X = []
        for data in iterator:
            inputs, targets  = data

            inputs = to_torchvar(inputs)
            targets= to_torchvar(targets)

            normed, z_content, z_style, z_normed = self.model(inputs, targets)
            X.append(normed.cpu().data.numpy())

        return np.concatenate(X, axis=0)
    
    def save(self, fname):
        torch.save(self.model, fname)
    
    def load(self, fname):
        self.model = torch.load(fname)