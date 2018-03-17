import smpl_tf
import smpl_np
import numpy as np
import tensorflow as tf


def compute_diff(a, b):
    return np.max(np.abs(a - b) / np.minimum(a, b))


def tf_wrapper(betas, pose, trans):
    betas = tf.constant(betas, dtype=tf.float64)
    trans = tf.constant(trans, dtype=tf.float64)
    pose = tf.constant(pose, dtype=tf.float64)
    output, _ = smpl_tf.smpl_model('./model.pkl', betas, pose, trans)
    sess = tf.Session()
    result = sess.run(output)
    return result


def np_wrapper(betas, pose, trans):
    result, _ = smpl_np.smpl_model('./model.pkl', betas, pose, trans)
    return result


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    betas = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    tf_result = tf_wrapper(betas, pose, trans)
    np_result = np_wrapper(betas, pose, trans)

    print(compute_diff(tf_result, np_result))
