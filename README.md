# Convolutional DRAW

Implementation of [Convolutional DRAW](https://arxiv.org/pdf/1604.08772.pdf) on MNIST, SVHN and CIFAR-10 in TensorFlow.

Here are some examples of reconstructions produced by the model:

| CIFAR-10  | SVHN |
| ------------- | ------------- |
| <img src="https://raw.githubusercontent.com/kuleshov/convolutional-draw/master/img/reconstructions.cifar10.3600.png" width="100%"> | <img src="https://raw.githubusercontent.com/kuleshov/convolutional-draw/master/img/reconstructions.svhn.8300.png" width="100%"> |

## Usage

You just need to run the python script for a given dataset. For example, `python draw-svhn.py` downloads the `svhn` dataset and trains the convolutional DRAW model. After training, output data is written to `/tmp/draw/draw_data.npy`; during training we generate reconstructions every 100 iterations.

## Acknowledgements

This code is based on the implementation of vanilla DRAW by Eric Jang.

## Feedbeck

Send feedback to [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/).

