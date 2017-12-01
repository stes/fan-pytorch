import numpy as np
from sklearn.decomposition import PCA
import h5py
from sklearn.preprocessing import LabelBinarizer

import torch
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['PCAIterator']


def flip(xx, flip_h=True, flip_v=True):
    """ Randomly flip images in a given batch
    """

    if flip_h:
        flip_h = np.random.choice([-1, 1], size=len(xx))
    else:
        flip_h = np.ones(len(xx))
    if flip_v:
        flip_v = np.random.choice([-1, 1], size=len(xx))
    else:
        flip_v = np.ones(len(xx))

    # TODO there a probably faster ways to do individual flipping using some indexig
    # technique?
    return np.stack((x[:, ::h, ::v] for x, h, v in zip(xx, flip_h, flip_v)), axis=0)


class PCAIterator():
    """ Augment samples along their principal components
    """

    def __init__(self, X, y=None):
        X_ = X[..., ::8, ::8].transpose((0, 2, 3, 1)).reshape(-1, 3)
        pca = PCA(n_components=3, copy=True, whiten=False)
        pca.fit(X_)
        self.W = pca.components_
        self.std = np.sqrt(pca.explained_variance_)
        self.W_inv = np.linalg.inv(self.W)
        self.shape = X.shape
        self.y = y


        assert np.allclose(np.dot(self.W, self.W_inv), np.eye(3), atol=1e-7)
        assert (y is None) or (len(y) == len(X))

    def sample_noise(self,batch_size,spatial_scale=1, noise_level=.5):
        shape = [batch_size,
                 int(self.shape[2]//spatial_scale),
                 int(self.shape[3]//spatial_scale)]

        noise1 = np.random.normal(0, self.std[0]*noise_level, size=shape)
        noise2 = np.random.normal(0, self.std[1]*noise_level, size=shape)
        noise3 = np.random.normal(0, self.std[2]*noise_level, size=shape)

        # stacked noise with [B, C', I, J] tdot(1) [C, C'] --> [B, I, J, C] -- transposed --> [B, C, I, J]
        n = np.tensordot(np.stack((noise1, noise2, noise3), axis=1), self.W_inv, axes=((1,) ,(1,))).transpose((0, 3, 1, 2))
        return F.upsample(Variable(torch.from_numpy(n)), None, spatial_scale, 'bilinear').data.numpy().squeeze()

    def iterate(self, X, y=None, batch_size=16, shuffle=True, flip_h=True, flip_v=True, augment=True, binarize = False):
        assert (y is None) or (len(y) == len(X))

        if binarize and (y is not None):
            lb = LabelBinarizer()
            y = lb.fit_transform(y)

        if shuffle:
            idc = np.arange(len(X))
            np.random.shuffle(idc)
            X = X[idc]
            if y is not None: y = y[idc]
                
        for idc in range(0, len(X), batch_size):
            if idc+batch_size > len(X): continue
            bX = flip(X[idc:idc+batch_size], flip_h, flip_v)
            if y is not None: by = y[idc:idc+batch_size]

            if augment:
                N = self.sample_noise(batch_size, spatial_scale=X.shape[3], noise_level=.9)
                N += self.sample_noise(batch_size, spatial_scale=X.shape[3]//4, noise_level=.8)
                N += self.sample_noise(batch_size, spatial_scale=X.shape[3]//16, noise_level=.1)
                d = np.clip(np.random.uniform(0.3,1.9,size=(bX.shape[0],1,1,1)),0,1)
            else:
                N = 0
                d = 1

            if y is not None:
                yield np.float32(np.clip(d*bX + N, 0, 1.)), np.float32(bX), by
            else:
                yield np.float32(np.clip(d*(bX + N), 0, 1.)), np.float32(bX)
