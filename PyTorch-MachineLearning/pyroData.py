import gzip
import pickle
import h5py
import numpy as np
import os

def shuffle_dset_labels(X, y):
    idc = np.random.permutation(list(range(X.shape[0])))
    return X[idc], y[idc]


def load_mnist():
    with gzip.open('/work/dbug/data/mnist.pkl.gz', 'rb') as f:
        upi = pickle._Unpickler(f)
        upi.encoding = 'latin1'
        data = upi.load()
    return data


def load_stainnorm_imgs():
    import joblib
    if os.path.exists('sn_data.gz'):
        return joblib.load('sn_data.gz')
    else:
        from glob import glob
        from skimage.io import imread
        imgs = {}
        for fname in glob('/images/ILLUMINATE/StainNormalization/final/ilu_*/ilu_*.tif'):
            imgs[fname] = np.float32(imread(fname)[:, :, :3])
            print(fname, np.max(imgs[fname]))
        joblib.dump(imgs, 'sn_data.gz', compress=4, cache_size=512, protocol=-1)
        return imgs


def cross_validate_mnist():
    # TODO: implement 5-10x CV for MNIST Dataset
    pass


def load_ilu(labels=['tumor',
                     'MouseStroma',
                     'connective tissue',
                     'Necrosis',
                     'Vacuola',
                     'blood vessel',
                     # 'Muscle',
                     # 'technical artefact',
                     'background'
                    ], verbose=False):

    def print_dict(data):
        for sl, grp in data.items():
            for lb, dset in grp.items():
                print(sl, lb, dset.shape)

    data = dict()
    with h5py.File('/work/dbug/data/ILU-9_cross_valid_ready.hdf5', 'r') as f:
        for skey, slide in f.items():
            data[skey] = dict()
            for lkey, dset in slide.items():
                if lkey in labels:
                    data[skey][lkey] = dset[...]
                else:
                    print((skey, lkey), 'was skipped')

            # merge CT into MST:
            if 'connective tissue' in data[skey].keys():
                if 'MouseStroma' in data[skey].keys():
                    data[skey]['MouseStroma'] = np.concatenate(
                        (data[skey]['MouseStroma'], data[skey]['connective tissue']),
                        axis=0)
                else:
                    data[skey]['MouseStroma'] = data[skey]['connective tissue']
                del data[skey]['connective tissue']

    if verbose:
        print_dict(data)
        
    return data


def get_dataset(fname="/work/dbug/dbug_ml/stainnorm_big_l1_p192.hdf5", train_key="H.E.T.", val_key="H.E.T+",
                val_slides=("47453", "74235")):
    to_list = lambda x: x if isinstance(x, list) else [x]
    train_key = to_list(train_key)
    val_key = to_list(val_key)
    X = []
    Xv = []
    with h5py.File(fname, "r") as ds:
        for key in list(ds.keys()):
            if not key in val_slides:
                for lbl in train_key:
                    X.append(ds[key][lbl][...])
            else:
                for lbl in val_key:
                    Xv.append(ds[key][lbl][...])
    X = np.concatenate(X, axis=0)
    Xv = np.concatenate(Xv, axis=0)
    np.random.shuffle(X)
    return np.float32(X), np.float32(Xv)



def cross_validate_ilu(data_in, labels=None):

    if labels is None:
        labels = {
            'tumor': 0,
            'MouseStroma': 1,
            'Necrosis': 2,
            'Vacuola': 3,
            'blood vessel': 4,
            'background': 5
        }

    def create_dset_labels():
        X = []
        y = []
        for cls, dset in data_in[key].items():
            X.append(dset)
            y.append(np.ones(dset.shape[:1], dtype=np.uint8)*labels[cls])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        return X, y

    N = len(data_in.keys())
    keys = list(data_in.keys())
    for idx in range(N):
        test_key = keys[idx]
        valid_key = keys[(idx+1) % N]

        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        X_test = []
        y_test = []

        for key in data_in.keys():
            if key == test_key:
                X_test, y_test = create_dset_labels()
                print('Using {} for testing'.format(key))
            elif key == valid_key:
                X_valid, y_valid = create_dset_labels()
                print('Using {} for validation'.format(key))
            else:
                X, y = create_dset_labels()
                X_train.append(np.copy(X))
                y_train.append(np.copy(y))
                print('Using {} for training'.format(key))

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        X_train, y_train = shuffle_dset_labels(X_train, y_train)
        X_valid, y_valid = shuffle_dset_labels(X_valid, y_valid)
        X_test, y_test = shuffle_dset_labels(X_test, y_test)

        data = (
            (X_train, y_train),
            (X_valid, y_valid),
            (X_test, y_test),
        )
        yield data


if __name__ == '__main__':

    X = load_stainnorm_imgs()

    print(X.shape)
