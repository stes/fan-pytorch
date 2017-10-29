"""
Module implements a combined MSELoss Autoencoder (Single Layer, Dense, Linear
Units) and CCELoss classifier.
"""

import torch
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from pyroMance import skrew_labels
from pyroModel import DenseAEC
from pyroData import load_mnist, load_ilu, cross_validate_ilu
from pyroTrain import train, test, plot_examples_mnist
import matplotlib.pyplot as plt

SAVE_DIR = '/home/temp/bug/resultsDump/'


def mlb_labels(y_train, y_valid, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit(y_train)
    y_train = mlb.transform(y_train[:, np.newaxis])
    y_valid = mlb.transform(y_valid[:, np.newaxis])
    y_train = skrew_labels(y_train, fraction=0.9)
    print('# zero labels', np.sum(0 == np.sum(y_train, axis=1)))
    return y_train, y_valid




def correlation_test(Xtr, Xva, Xte, ytr, yva, yte):

    def get_histogram(img):
        hist = []
        for c in range(img.shape[0]):
            h, _ = np.histogram(img[c], bins=np.arange(35, 251, 5))
            hist.append(h)
        return np.concatenate(hist, axis=0)

    Hva = []
    Hte = []
    for i in range(Xva.shape[0]):
        Hva.append(get_histogram(Xva[i]))
    for i in range(Xte.shape[0]):
        Hte.append(get_histogram(Xte[i]))

    va_images = []
    te_images = []
    va_captions = []
    te_captions = []
    for _ in range(10):
        idx = np.random.randint(0, Xtr.shape[0])
        print('Using index', idx)
        va_images.append(np.copy(Xtr[idx]))
        te_images.append(np.copy(Xtr[idx]))
        va_captions.append(ytr[idx])
        te_captions.append(ytr[idx])

        h = get_histogram(Xtr[idx])

        va_similarities = []
        for va in range(Xva.shape[0]):
            va_similarities.append(np.sum(np.abs(h - Hva[va])))

        te_similarities = []
        for te in range(Xte.shape[0]):
            te_similarities.append(np.sum(np.abs(h - Hte[te])))

        va_smallest = np.argsort(va_similarities)[:5]
        te_smallest = np.argsort(te_similarities)[:5]
        plt.figure(0)
        for jj in range(5):
            va_images.append(Xva[va_smallest[jj]])
            va_captions.append(yva[va_smallest[jj]])
            te_images.append(Xte[te_smallest[jj]])
            te_captions.append(yte[te_smallest[jj]])

    plt.figure(0, figsize=(9, 5))
    for ii in range(len(va_images)):
        plt.subplot(10, 6, ii+1)
        plt.imshow(va_images[ii].swapaxes(0, 2))
        plt.title(str(va_captions[ii]))
        plt.xticks(())
        plt.yticks(())

    plt.figure(1, figsize=(9, 5))
    for ii in range(len(te_images)):
        plt.subplot(10, 6, ii+1)
        plt.imshow(te_images[ii].swapaxes(0, 2))
        plt.title(str(te_captions[ii]))
        plt.xticks(())
        plt.yticks(())
    plt.show()


if '__main__' == __name__:

    torch.cuda.device(0)

    ilu_data = load_ilu()
    for data in cross_validate_ilu(ilu_data):
        Xtr, ytr = data[0]
        Xva, yva = data[1]
        Xte, yte = data[2]
        print('train:', Xtr.shape, ytr.shape)
        print('valid:', Xva.shape, yva.shape)
        print(' test:', Xte.shape, yte.shape)
        correlation_test(Xtr, Xva, Xte, ytr, yva, yte)


    data = load_mnist()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    y_train, y_valid = mlb_labels(y_train, y_valid)
    data = (
        (X_train, y_train),
        (X_valid, y_valid),
        (X_test, y_test),
    )
    model = DenseAEC(784, 32, output_size=10)
    model.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    train(100, data, model, optimizer,
          mse_weight=1.0,
          cce_weight=0.3,
          preview_plot=plot_examples_mnist, savepath=SAVE_DIR+'training_mixf03_n32.png')
    torch.save(model, 'denseAE_mixf03_n32.t7')
    test(data, model, preview_plot=plot_examples_mnist, savepath=SAVE_DIR+'test_mixf03_n32.png')
    print('__DONE__')
