import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from pyroMance import batch_iterator, SelectiveCrossEntropyLoss, skrew_labels
from pyroAugment import PCAIterator
from pyroData import get_dataset, load_stainnorm_imgs
import logging as log


def plot_examples_mnist(reconstruction, truth, prediction, nr):
    out = reconstruction.cpu().data.numpy()
    pred = np.argmax(prediction.cpu().data.numpy(), axis=1)
    plt.figure(1).clear()
    lim = min(out.shape[0], nr)
    s = np.uint8(np.ceil(np.sqrt(lim)))
    for i in range(lim):
        plt.subplot(s, s, i+1)
        plt.imshow(np.reshape(out[i, :], (28, 28)))
        plt.xticks(())
        plt.yticks(())
        plt.title('t:{}, p:{}'.format(truth[i], pred[i]))
    plt.tight_layout(pad=0.02)
    plt.draw()
    plt.pause(1)


def tile_plot(reconstruction, nr):
    out = reconstruction.cpu().data.numpy()
    print(out.shape, out.max())
    plt.figure(1).clear()
    lim = min(out.shape[0], nr)
    s = np.uint8(np.ceil(np.sqrt(lim)))
    for i in range(lim):
        plt.subplot(s, s, i+1)
        plt.imshow(np.clip(np.uint8(out[i, :, :, :]), 0, 255).swapaxes(0, 2))
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout(pad=0.01)
    plt.draw()
    plt.pause(5)


def train(epoch, model, optimizer, mse_weight=1., cce_weight=1., preview_plot=None, savepath=''):

    log.info('Loading Dataset')
    X, Xv = get_dataset()
    Xv = Xv[::4]

    log.info('Generating PCA Augmentation')
    generator = PCAIterator(np.concatenate((X, Xv), axis=0))
    
    mse = nn.MSELoss()

    plt.ion()
    losses_mse = []
    minLoss = 1e100
    log.info('Start Training')
    for ep in range(epoch):
        # start with validation loss as baseline
        model.train(False)
        log.info('Epoch: {:d}'.format(ep))
        ep_losses_mse = []
        acc = 0
        for src, tgt in generator.iterate(Xv, batch_size=24, shuffle=True):
            batch = Variable(torch.from_numpy(src).cuda())
            target = Variable(torch.from_numpy(tgt).cuda())
            normed, features = model(batch)
            loss_mse = mse(normed, target)
            ep_losses_mse.append(loss_mse.cpu().data.numpy())
        L = np.mean(ep_losses_mse)
        log.info(' o VALID losses: mse {}'.format(L))
        losses_mse.append(L)
        
        if minLoss > L and ep > epoch//20:
            minLoss = L
            torch.save(model, 'trained_models/tissue_sn_ep{:03d}.t7'.format(ep))
            log.info('   ...new model saved.')
        
        # train the network
        model.train(True)
        for src, tgt in generator.iterate(X, batch_size=24, shuffle=True):
            optimizer.zero_grad()
            batch = Variable(torch.from_numpy(src).cuda())
            target = Variable(torch.from_numpy(tgt).cuda())
            normed, features = model(batch)
            loss = mse(normed, target)
            loss.backward()
            optimizer.step()
        log.info(' - Loss: {}'.format(loss.cpu().data.numpy()))


def test(model, preview_plot=None, savepath=''):
    X, Xv = get_dataset()
    Xv = Xv[::4] 
    print(np.max(Xv))
    generator = PCAIterator(np.concatenate((X, Xv), axis=0))

    mse = nn.MSELoss()
    model.train(False)
    losses = []

    first_run = 0
    for src, tgt in generator.iterate(Xv, batch_size=16, shuffle=False):
        batch = Variable(torch.from_numpy(src).cuda())
        target = Variable(torch.from_numpy(tgt).cuda())
        reconstruction, features = model(batch)
        loss_mse = mse(reconstruction, target)
        losses.append(loss_mse.cpu().data.numpy())
        if first_run<3 and preview_plot is not None and savepath != '':
            first_run += 1
            plt.close()
            preview_plot(reconstruction, 16)
            plt.savefig(savepath.format(first_run), format='png', dpi=120, pad_inches=0.1, bbox_inches='tight')
            plt.close()
    print('(-: TEST loss:  mse {}'.format(np.mean(losses)))


def normalize_sn_imgs(model):
    from skimage.io import imsave

    def crop(img, delta=(8, 8)):
        dx = delta[0]//2
        dy = delta[1]//2
        du = (delta[0]//2) + delta[0]%2
        dv = (delta[1]//2) + delta[1]%2
        img_c = img[dx:-du, dy:-dv]

        return img_c

    D = load_stainnorm_imgs()
    for fname, img in D.items():
        img_c = crop(img)
        img_b = np.expand_dims(img_c, 0)
        img_b = img_b.swapaxes(1, 3)
        img_b = Variable(torch.from_numpy(img_b).cuda())

        normed, features = model(img_b)
        
        img_b = normed.cpu().data.numpy()
        img_b = img_b.swapaxes(1, 3)
        img_b = np.squeeze(img_b)

#        plt.subplot(121)
#        plt.imshow(np.uint8(img))
#        plt.subplot(122)
#        plt.imshow(np.uint8(img_b))
#        plt.show()
        savenm = fname.replace('_aligned', '_normed')
        savenm = savenm.replace('.tif', '.jpg')
        savenm = savenm.replace('/images/ILLUMINATE/StainNormalization/final', 'results')
        print(savenm)

        imsave(savenm, np.clip(img_b/255., .0, 1.))
