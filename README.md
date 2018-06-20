# fan-pytorch
## *NOTE: Repository still under construction*

Feature Aware Normalization
Project homepage: [stes.github.io/fan](https://stes.github.io/fan)

Implementation of Feature Aware Normalization in PyTorch.
Original implementation with Demo in Theano: [github.com/stes/fan](https://github.com/stes/fan)

## Interfaces

A full network for color normalization can be accessed either directly using the PyTorch module or by using a `sklearn`-like interface:

```
from fan.stainnorm import StainNormalizerMultiFAN
```

## Dependencies

Code was tested using Python 3.6.

To run the FAN demo, please make sure that you installed the following packages:
- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `h5py`
- `sklearn`

```
pip -r requirements.txt
```


## Dataset

We also provide the dataset used for the validation of our approach.
Please refer to the [project homepage](https://stes.github.io/fan) to download the dataset.

## Reference

```
@incollection{bug2017context,
  title={Context-based Normalization of Histological Stains using Deep Convolutional Features},
  author={Bug, Daniel and Schneider, Steffen and Grote, Anne and Oswald, Eva and Feuerhake, Friedrich and Sch{\"u}ler, Julia and Merhof, Dorit},
  booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support},
  pages={135--142},
  year={2017},
  publisher={Springer}
}
```
