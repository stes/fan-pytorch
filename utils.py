from collections import OrderedDict
import numpy as np

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