import smpl_tf
import smpl_np
import numpy as np
import tensorflow as tf


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


def tf_wrapper(beta, pose, trans):
    beta = tf.constant(beta, dtype=tf.float64)
    trans = tf.constant(trans, dtype=tf.float64)
    pose = tf.constant(pose, dtype=tf.float64)
    output, _ = smpl_tf.smpl_model('./model.pkl', beta, pose, trans)
    sess = tf.Session()
    result = sess.run(output)
    return result


def np_wrapper(beta, pose, trans):
    smpl = smpl_np.SMPLModel('./model.pkl')
    result = smpl.set_params(pose=pose, beta=beta, trans=trans)
    return result


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    beta = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    tf_result = tf_wrapper(beta, pose, trans)
    np_result = np_wrapper(beta, pose, trans)

    if np.allclose(np_result, tf_result):
        print('Bingo!')
    else:
        print('Failed')
        print(compute_diff(tf_result, np_result))
