# SMPL
Numpy and Tensorflow implementation of SMPL model. For any questions, feel free to contact me.

## Overview

I wrote this because the author-provided implementation was mainly based on [chumpy](https://github.com/mattloper/chumpy) in Python 2, which is kind of unpopular. Meanwhile, the official one cannot run on GPU.

This numpy version is faster(since some computation is re-wrote in a vectorized manner) and easier to understand(hope so), and the tensorflow version can run on GPU.

For more details about SMPL model, see [SMPL](http://smpl.is.tue.mpg.de/).

Also, I provide a file `CMU_Mocap_Markers.pp`, which gives the correspondence between SMPL model and [CMU Mocap Dataset](http://mocap.cs.cmu.edu/) markers in .c3d files. For more detailes see the Usage section.

## Usage

1. Download the model file [here](http://smpl.is.tue.mpg.de/downloads).
2. Run `python preprocess.py /PATH/TO/THE/DOWNLOADED/MODEL` to preprocess the official model. `preprocess.py` will create a new file `model.pkl`. `smpl_np.py` and `smpl_tf.py` both rely on `model.pkl`. NOTE: the official pickle model contains `chumpy` object, so `prerocess.py` requires `chumpy` to extract official model.
3. Run `python smpl_np.py` to see the example.
4. About `CMU_Mocap_Markers.pp`: you can first generate a standard SMPL model mesh(zero pose and zero beta), open it in MeshLab, and load this file in MeshLab. It gives 42 markers' position on the model surface. I simply mark these things by hand so there might be some small errors.

## One More Thing

If this repo is used in any publication or project, it would be nice to let me know. I will be very happy and encouraged =)
