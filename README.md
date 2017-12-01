# fan-pytorch

Feature Aware Normalization
Project homepage: [stes.github.io/fan](https://stes.github.io/fan)

Implementation of Feature Aware Normalization in PyTorch.

## Interfaces

A full network for color normalization can be accessed either directly using the PyTorch module or by using a `sklearn`-like interface:

```
from fan.stainnorm import StainNormalizerMultiFAN
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
