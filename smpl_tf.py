import tensorflow as tf
import numpy as np
import pickle


def rodrigues_single(r):
    theta = tf.norm(r)
    theta = tf.maximum(theta, 1e-8)
    r_hat = tf.reshape(r / theta, (3, 1))
    cos = tf.cos(theta)
    z = tf.zeros((1), dtype=tf.float64)
    m = tf.reshape(
        tf.stack((z, -r_hat[2], r_hat[1], r_hat[2], z, -r_hat[0], -r_hat[1],
                  r_hat[0], z)), (3, 3))
    R = cos * tf.eye(3, dtype=tf.float64) + (
        1 - cos) * r_hat * tf.transpose(r_hat) + tf.sin(theta) * m
    return R


def rodrigues(r):
    theta = tf.norm(r, axis=(0, 1))
    theta = tf.maximum(theta, 1e-8)
    r_hat = r / theta
    cos = tf.cos(theta)
    z_stick = tf.zeros(theta.get_shape()[0], dtype=tf.float64)  # simply use static shape here
    m = tf.stack(
        (z_stick, -r_hat[0, 2, :], r_hat[0, 1, :], r_hat[0, 2, :], z_stick,
         -r_hat[0, 0, :], -r_hat[0, 1, :], r_hat[0, 0, :], z_stick))
    m = tf.reshape(m, (3, 3, -1))
    i_cube = tf.reshape(tf.eye(3, dtype=tf.float64), (3, 3, 1)) + tf.zeros(
        (3, 3, theta.get_shape()[0]), dtype=tf.float64)
    A = tf.transpose(r_hat, (2, 1, 0))
    B = tf.transpose(r_hat, (2, 0, 1))
    dot = tf.transpose(tf.matmul(A, B), (1, 2, 0))
    R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m
    return R


def with_zeros(x):
    return tf.concat((x, tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float64)), axis=0)


def pack(x):
    depth = x.get_shape().as_list()[-1]
    z_cube = tf.zeros((4, 3, depth), dtype=tf.float64)
    x_stick = tf.reshape(x, (4, 1, depth))
    return tf.concat((z_cube, x_stick), axis=1)


def smpl_model(model_path, betas, pose, trans):
    with open(model_path, 'rb') as f:
        params = pickle.load(f)

    J_regressor = tf.constant(
        np.array(params['J_regressor'].todense(), dtype=np.float64))
    weights = tf.constant(params['weights'], dtype=np.float64)
    posedirs = tf.constant(params['posedirs'], dtype=np.float64)
    v_template = tf.constant(params['v_template'], dtype=np.float64)
    shapedirs = tf.constant(params['shapedirs'], dtype=np.float64)
    f = params['f']

    kintree_table = params['kintree_table']
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {
        i: id_to_col[kintree_table[0, i]]
        for i in range(1, kintree_table.shape[1])
    }

    v_shaped = tf.tensordot(shapedirs, betas, axes=[[2], [0]]) + v_template

    J = tf.matmul(J_regressor, v_shaped)

    pose_cube = tf.transpose(tf.reshape(pose[3:], (-1, 1, 3)), (1, 2, 0))

    R_cube = rodrigues(pose_cube)
    I_cube = tf.reshape(tf.eye(3, dtype=tf.float64), (3, 3, 1)) + tf.zeros(
        (3, 3, R_cube.get_shape()[2]), dtype=tf.float64)
    lrotmin = tf.squeeze(
        tf.reshape(tf.transpose(R_cube - I_cube, (2, 0, 1)), [1, -1]))

    v_posed = v_shaped + tf.tensordot(posedirs, lrotmin, axes=[[2], [0]])

    pose = tf.reshape(pose, (-1, 3))

    depth = kintree_table.shape[1]

    results = []

    results.append(
        with_zeros(
            tf.concat(
                (
                    rodrigues_single(pose[0, :]),
                    tf.reshape(
                        J[0, :],
                        (3, 1)
                    )
                ),
                axis=1
            )
        )
    )

    for i in range(1, kintree_table.shape[1]):
        results.append(
            tf.matmul(
                results[parent[i]],
                with_zeros(
                    tf.concat(
                        (
                            rodrigues_single(pose[i, :]),
                            tf.reshape(J[i, :] - J[parent[i], :], (3, 1))
                        ),
                        axis=1
                    )
                )
            )
        )

    stacked = tf.stack(results, axis=2)

    results = stacked - pack(
        tf.transpose(
            tf.matmul(
                tf.transpose(stacked, (2, 0, 1)),
                tf.reshape(
                    tf.concat(
                        (
                            J,
                            tf.zeros((24, 1), dtype=tf.float64)
                        ),
                        axis=1
                    ),
                    (24, 4, 1)
                )
            ),
            (1, 2, 0)
        )
    )

    T = tf.tensordot(results, weights, axes=[[2], [1]])

    rest_shape_h = tf.concat((tf.transpose(v_posed),
                             tf.ones((1, v_posed.get_shape()[0]), dtype=tf.float64)), axis=0)

    v = tf.matmul(
        tf.transpose(T, (2, 0, 1)),
        tf.transpose(
            tf.reshape(
                rest_shape_h,
                (4, 1, -1)
            ),
            (2, 0, 1)
        )
    )

    v = tf.reshape(v, (-1, 4))[:, :3]

    result = v + tf.reshape(trans, (1, 3))
    return result, f


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    betas = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    pose = tf.constant(pose, dtype=tf.float64)
    betas = tf.constant(betas, dtype=tf.float64)
    trans = tf.constant(trans, dtype=tf.float64)

    output, faces = smpl_model('./model.pkl', betas, pose, trans)

    sess = tf.Session()

    result = sess.run(output)

    outmesh_path = './hello_smpl.obj'
    with open(outmesh_path, 'w') as fp:
        for v in result:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
