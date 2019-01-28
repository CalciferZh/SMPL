# SMPL
Numpy, Tensorflow and PyTorch implementation of SMPL model. For any questions, feel free to contact [me](https://github.com/CalciferZh/SMPL).

## Overview

### Update on 20190127 by [Lotayou](https://github.com/Lotayou)
I write a PyTorch implementation based on CalciferZh's Tensorflow code, which supports batch processing and GPU training. The implementation is hosted in `smpl_torch.py` along with the testing example.

The implementation is tested under Ubuntu 18.04, Python 3.6 and Pytorch 1.0.0 stable. The output is the same as the original Tensorflow implementation, as can be tested with `test.py`.

#### Original Overview
I wrote this because the author-provided implementation was mainly based on [chumpy](https://github.com/mattloper/chumpy) in Python 2, which is kind of unpopular. Meanwhile, the official version cannot run on GPU.

This numpy version is faster (since some computations were rewrote in a vectorized manner) and easier to understand (hope so), and the TensorFlow version can run on GPU.

For more details about SMPL model, see [SMPL](http://smpl.is.tue.mpg.de/).

## Usage

1. Download the model file [here](http://smpl.is.tue.mpg.de/downloads).
2. Run `python preprocess.py /PATH/TO/THE/DOWNLOADED/MODEL` to preprocess the official model. `preprocess.py` will create a new file `model.pkl`. `smpl_np.py` and `smpl_tf.py` both rely on `model.pkl`. **NOTE**: the official pickle model contains `chumpy` object, so `prerocess.py` requires `chumpy` to extract official model. You need to modify chumpy's cource code a little bit to make it compatible to `preprocess.py` (and Python 3). [Here](https://blog.csdn.net/qq_28660035/article/details/81319055) is an instruction in Chinese about this.
If you don't want to install `chumpy`, you can download processed file from [BaiduYunDisk](https://pan.baidu.com/s/1TqKitzV-EtIOowN0xwpQng) with extraction code `vblg`

3. Run `python smpl_np.py` or `python smpl_tf.py` or `python smpl_torch.py` to see the example. Additionally, run `python smpl_torch_batch.py` for batched support.
