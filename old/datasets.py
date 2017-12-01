import h5py
import numpy as np
import os
from skimage import io
from tifffile import imread

def get_iluminate(fname="data/patches_train_patches_192.hdf5",
                train_key="H.E.T+", val_key="H.E.T.",
                val_slides=["47453", "74235"]):
    to_list = lambda x : x if isinstance(x, list) else [x]
    train_key = to_list(train_key)
    val_key = to_list(val_key)
    X = []
    Xv = []
    with h5py.File(fname, "r") as ds:
        for key in list(ds.keys()):
            if not key in val_slides:
                for lbl in train_key:
                    X.append(ds[key][lbl][:,...])
            else:
                for lbl in val_key:
                    Xv.append(ds[key][lbl][:,...])
    X = np.concatenate(X, axis=0)
    Xv = np.concatenate(Xv, axis=0)
    np.random.shuffle(X)
    return np.float32(X), np.float32(Xv)

def get_kather(root):
    classes = os.listdir(root)
    X     = []
    y     = []
    lbl   = []
    fnames = []
    
    for i, cl in enumerate(sorted(classes)):
        classdir = os.path.join(root, cl)
        if not os.path.isdir(classdir): continue
        imgfiles = os.listdir(classdir)
        
        for j, fname in enumerate(imgfiles):
            path = os.path.join(root, cl, fname)
            X.append(imread(path))
            y.append(i)
            fnames.append(path)
        lbl.append(cl)
        
    X = np.stack(X, axis=0)[...,:3].transpose((0,3,1,2))
    y = np.stack(y, axis=0)
    
    return X, y, lbl, fnames