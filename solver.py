""" Demo dataset to train a model for feature aware normalization
"""
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import torch
from torch import nn
import time
from torch.autograd import Variable

from fan import datasets, pca, utils, stainnorm

CUDA_DEVICE_ID = 1

#### Helper functions ####

def cuda(arg):
    if CUDA_DEVICE_ID is not None:
        return arg.cuda(CUDA_DEVICE_ID)

def upsample(x):
    """ bilinear upsampling to patchsize 320px """
    return F.upsample(Variable(torch.from_numpy(x)), (256+64,256+64), None, 'bilinear').data.numpy()

def to_torchvar(x):
    x = Variable(torch.from_numpy(x).float())
    return x.cuda(device_id)

#### Experiment setup ####

def load_data():
    """ load iluminate training dataset """
    X, Xt = datasets.get_iluminate('../fan-theano/data/train_patches_192_l0.hdf5',
                                  train_key=["H.E.T+"])

    # patches have 512px resolution and will be further preprocessed
    X = X / 255.
    Xt = Xt / 255.

    # extract 192px patches out of the original dataset
    Xs = np.concatenate([
            X[...,i*192:(i+1)*192,j*192:(j+1)*192] for i in range(2) for j in range(2)
        ], axis=0)

    # to estimate a better noise model, also load the Kather dataset
    Xk, yk, _, _ = datasets.get_kather('data/Kather_texture_2016_image_tiles_5000')
    Xk = upsample(Xk / 255.)

def load_model(pretrained=None):
    if pretrained is not None:
        model = torch.load(pretrained)
    model = stainnorm.StainNormalizerMultiFAN()
    cuda(model)

#### Visualization ####

for x,y in iterator.iterate(Xs, augment=True):
    plt.imshow(utils.panelize(x[0:10]))
    plt.show()
    plt.imshow(utils.panelize(y[0:10]))
    plt.show()
    break

#### Optimization ####


def experiment():
    load_data()
    model = load_model()
    iterator = pca.PCAIterator(np.concatenate([Xs[::20,:,:,:], Xk[yk<=5,:,:192,:192]], axis=0))

def load_optimizer():
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.get_training_params(), lr = 1e-3)
    cuda(loss)

    return loss, optimizer

def train(X, model, iterator, loss, optimizer):
    """ Training loop
    """
    losses_collect = []
    ref_losses = []

    model.train(True)
    tic = time.time() - 60*20
    for n in range(n_epochs):
        for data in iterator.iterate(X,augment=True, batch_size=16,
                                     flip_h=True, flip_v=True, shuffle=True):
            inputs, targets  = data

            inputs  = to_torchvar(inputs)
            targets = to_torchvar(targets)

            optimizer.zero_grad()

            normed = model(inputs, targets)
            losses = loss(normed, targets)

            sum(losses).backward()
            optimizer.step()

            losses_collect.append( [l.cpu().data[0] for l in losses] )
            ref_losses.append( [loss(inputs, targets).cpu().data[0] for l in losses] )

            toc = time.time()
            if (toc - tic > 60*20) and (len(losses_collect) > 20):
                loss_mean = -10 * np.log10(np.array(losses_collect)[-20:].mean(axis=0))
                ref_mean  = -10 * np.log10(np.array(ref_losses)[-20:].mean(axis=0))
                print('Content {:.3f}dB (Intially {:.3f})'.format(*loss_mean, *ref_mean))
                tic = time.time()

                z = normed.data.cpu().numpy()
                plt.imshow(utils.panelize(np.clip(z[0:10],0,1.)))
                plt.show()

                torch.save(model, 'normalizer_{}.pth'.format(time.time()))

#### Application ####

def normalize():
    """ apply normalization """
    iterator = pca.PCAIterator(X[::20,:,:,:])
    model.cuda(device_id)

    normed_imgs = []

    for data in iterator.iterate(X[...,:,:],augment=False, batch_size=20, flip_h=True, flip_v=True, shuffle=False):
        inputs, targets  = data

        print(inputs.shape)
        inputs = (upsample(inputs.astype("float32")))[...,22:256+22,22:256+22]
        print(inputs.shape)

        x = inputs.copy()
        y = targets.copy()

        inputs  = to_torchvar(inputs)
        targets = to_torchvar(targets)

        normed = model(inputs, None)

        z = normed.data.cpu().numpy()

        normed_imgs.append(z)


if __name__ == '__main__':
    import sys
    import argparse
    from stainnorm.tools import logger_setup

    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument("-m", '--model', nargs="+",
        help='Model ID')
    parser.add_argument('-H', '--hematoxylin', type=int, nargs=1, default=1,
        help='control hematoxylin stain (-1/0/1)')
    parser.add_argument('-E', '--eosin', type=int, nargs=1, default=1,
        help='control eosin stain (-1/0/1)')
    parser.add_argument('-T', '--thickness', type=int, nargs=1, default=1,
        help='control thickness (-1/0/1)')
    parser.add_argument('-c', '--comment', type=str, nargs=1, default="run",
        help='comment string, will be added to the model name')
    args = parser.parse_args()
    mapping = ('-', '.', '+')
    dataset_code = "H{}E{}T{}".format(*[mapping[i[0]] for i in (args.hematoxylin, args.eosin, args.thickness)])

    print("Starting with dataset code {}".format(dataset_code))

    os.makedirs('log', exist_ok=True)
    logger_setup(filename="log/stainnorm.log")
    log = logging.getLogger(__name__)
    log.info("__START__")
