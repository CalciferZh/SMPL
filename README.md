# SMPL

Numpy, TensorFlow and PyTorch implementation of SMPL model. For C++ implementation (with PyTorch), please see this [repo](https://github.com/YeeCY/SMPLpp).

**Important:** I **do not** provide model file due to copyright reasons.

## Update Feb 2 2019
Now we have a faster PyTorch implementation, and we also support [SMIL](https://www.iosb.fraunhofer.de/servlet/is/82920/) model. For more details, please check [this](https://github.com/CalciferZh/SMPL/pull/11) PR.

## Overview

The author-provided implementation was mainly based on [chumpy](https://github.com/mattloper/chumpy) in Python 2, which is kind of unpopular. Meanwhile, the official version cannot run on GPU. This project provides Numpy, TensorFlow and PyTorch implementation of SMPL model.

For more details about SMPL model, see [SMPL](http://smpl.is.tue.mpg.de/).

### Numpy and Tensorflow Implementation

Contributor: [CalciferZh](https://github.com/CalciferZh).

The numpy version is faster (since some computations were rewrote in a vectorized manner) and easier to understand (hope so), and the TensorFlow version can run on GPU.

### PyTorch Implementation with Batch Input

Contributor: [Lotayou](https://github.com/Lotayou) and [sebftw](https://github.com/sebftw)

The PyTorch version is derived from the Tensorflow version, and in addition supports batch processing and GPU training. The implementation is hosted in `smpl_torch.py` along with the testing example.

The implementation is tested under Ubuntu 18.04, Python 3.6 and Pytorch 1.0.0 stable. The output is the same as the original Tensorflow implementation, as can be tested with `test.py`.

`SMIL_torch_batch.py` can be very fast, but limited by the memory. It also works with sparse tensors. (Saving a lot of said memory)

## Usage

1. Download the model file [here](http://smpl.is.tue.mpg.de/downloads).
2. Run `python preprocess.py /PATH/TO/THE/DOWNLOADED/MODEL` to preprocess the official model. `preprocess.py` will create a new file `model.pkl`. `smpl_np.py` and `smpl_tf.py` both rely on `model.pkl`. **NOTE**: the official pickle model contains `chumpy` object, so `prerocess.py` requires `chumpy` to extract official model. You need to modify chumpy's cource code a little bit to make it compatible to `preprocess.py` (and Python 3). [Here](https://blog.csdn.net/qq_28660035/article/details/81319055) is an instruction in Chinese about this.

3. Run `python smpl_np.py` or `python smpl_tf.py` or `python smpl_torch.py` to see the example. Additionally, run `python smpl_torch_batch.py` for batched support.
