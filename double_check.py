import smpl_tf
import smpl_np
import numpy as np
import tensorflow as tf


def compute_diff(a, b):
    return np.max(np.abs(a - b) / np.minimum(a, b))


def tf_wrapper(trans, betas, pose=None, R=None):
    betas = tf.constant(betas, dtype=tf.float32)
    trans = tf.constant(trans, dtype=tf.float32)
    if pose is not None:
        pose = tf.constant(pose, dtype=tf.float32)
        output, _ = smpl_tf.smpl_model('./model.pkl', trans, betas, pose=pose)
    elif R is not None:
        R = tf.constant(R, dtype=tf.float32)
        output, _ = smpl_tf.smpl_model('./model.pkl', trans, betas, R=R)
    else:
        return None
    sess = tf.Session()
    result = sess.run(output)
    return result


def np_wrapper(trans, betas, pose):
    result, _ = smpl_np.smpl_model('./model.pkl', trans, betas, pose)
    return result


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    betas = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    R = smpl_np.rodrigues(pose.reshape((-1, 1, 3)))

    tf_result = tf_wrapper(trans, betas, R=R)
    np_result = np_wrapper(trans, betas, pose)

    print(compute_diff(tf_result, np_result))
