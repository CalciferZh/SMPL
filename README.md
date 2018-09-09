# SMPL
Numpy and Tensorflow implementation of SMPL model. For any questions, feel free to contact me.

## Overview

I wrote this because the author-provided implementation was mainly based on [chumpy](https://github.com/mattloper/chumpy) in Python 2, which is kind of unpopular. Meanwhile, the official one cannot run on GPU.

This numpy version is faster (since some computation is re-wrote in a vectorized manner) and easier to understand (hope so), and the tensorflow version can run on GPU.

For more details about SMPL model, see [SMPL](http://smpl.is.tue.mpg.de/).

## Usage

1. Download the model file [here](http://smpl.is.tue.mpg.de/downloads).
2. Run `python preprocess.py /PATH/TO/THE/DOWNLOADED/MODEL` to preprocess the official model. `preprocess.py` will create a new file `model.pkl`. `smpl_np.py` and `smpl_tf.py` both rely on `model.pkl`. **NOTE**: the official pickle model contains `chumpy` object, so `prerocess.py` requires `chumpy` to extract official model. You need to modify chumpy's cource code a little bit to make it compatible to `preprocess.py` (and Python 3). [Here](https://blog.csdn.net/qq_28660035/article/details/81319055) is an instruction in Chinese about this.
3. Run `python smpl_np.py` or `python smpl_tf.py` to see the example.
