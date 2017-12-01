from collections import OrderedDict
import numpy as np

import numpy as np
import itertools as ito
import matplotlib.pyplot as plt
import json


def fixed_splits(fjson, fnames):
    with open(fjson, 'r') as fp:
        data = json.load(fp)

    assert fnames == data['fnames']

    for fold in data['folds']:
        # train_idx, test_idx = fold['train'], fold['test']
        yield fold['train'], fold['test']


def domain_splits(domainlbls):
    for i in range(10):
        train_idc = np.where((domainlbls != i))[0]
        test_idc = np.where((domainlbls == i))[0]
        yield train_idc, test_idc


def show_random_plot(X, y):
    idc = np.random.permutation(np.arange(0, 5000))

    for i in range(6 * 8):
        plt.subplot(6, 8, i + 1)
        plt.imshow(X[idc[i]], cmap='gray')
        plt.xticks(())
        plt.yticks(())
        plt.title(str(y[idc[i]]))
    plt.tight_layout(pad=0.05)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in ito.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    import h5py
    with h5py.File("X:\\temp\\ilu-p150-normed.hdf5", 'r') as ds:
        print(list(ds.keys()))

def panelize(img):
    if img.ndim == 1:
        raise ValueError("Invalid dimensions for image data" + str(img.shape))
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return panelize(img[np.newaxis, :, :])

    nb = img.shape[0]
    nb_rows = int(nb ** 0.5)
    psize = img.shape[2]
    nb_channel = img.shape[1]

    w, h = img.shape[-2:]

    img_per_row = nb // nb_rows
    rows = []
    for j in range(nb_rows):
        start = j * img_per_row
        stop = min(start + img_per_row, nb)
        rows.append(
            np.hstack([img[j, :, :, :].reshape(nb_channel, w, h).transpose((1, 2, 0)) for j in range(start, stop)]))
    return np.vstack(rows)
