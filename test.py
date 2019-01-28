import smpl_tf
import smpl_np
from smpl_torch import SMPLModel
import numpy as np
import tensorflow as tf
import torch
import os


def compute_diff(a, b):
    """
    Compute the max relative difference between ndarray a and b element-wisely.

    Parameters:
    ----------
    a, b: ndarrays to be compared of same shape.

    Return:
    ------
    The max relative difference.

    """
    return np.max(np.abs(a - b) / np.minimum(a, b))


def pytorch_wrapper(beta, pose, trans):
    device = torch.device('cuda')
    pose = torch.from_numpy(pose).type(torch.float64).to(device)
    beta = torch.from_numpy(beta).type(torch.float64).to(device)
    trans = torch.from_numpy(trans).type(torch.float64).to(device)
    model = SMPLModel(device=device)
    with torch.no_grad():
        result = model(beta, pose, trans)
    return result.cpu().numpy()


def tf_wrapper(beta, pose, trans):
    beta = tf.constant(beta, dtype=tf.float64)
    trans = tf.constant(trans, dtype=tf.float64)
    pose = tf.constant(pose, dtype=tf.float64)
    output, _ = smpl_tf.smpl_model('./model.pkl', beta, pose, trans)
    with tf.Session() as sess:
        result = sess.run(output)
    return result


def np_wrapper(beta, pose, trans):
    smpl = smpl_np.SMPLModel('./model.pkl')
    result = smpl.set_params(pose=pose, beta=beta, trans=trans)
    return result


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    beta = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    tf_result = tf_wrapper(beta, pose, trans)
    np_result = np_wrapper(beta, pose, trans)
    torch_result = pytorch_wrapper(beta, pose, trans)

    if np.allclose(np_result, tf_result) and np.allclose(np_result, torch_result):
        print('Bingo!')
    else:
        print('Failed')
        print('tf - np: ', compute_diff(tf_result, np_result))
        print('torch - np: ', compute_diff(torch_result, np_result))

