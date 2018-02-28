import numpy as np
import pickle
import cv2


def rodrigues_single(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    r_hat = (r / theta).reshape(3, 1)
    cos = np.cos(theta)
    m = np.array([
        0, -r_hat[2], r_hat[1], r_hat[2], 0, -r_hat[0], -r_hat[1], r_hat[0], 0
    ]).reshape((3, 3))
    R = cos * np.eye(3) + (1 - cos) * r_hat * r_hat.T + np.sin(theta) * m
    return R


def rodrigues(r):
    theta = np.linalg.norm(r, axis=(0, 1))
    r_hat = r / theta
    cos = np.cos(theta) # theta is radius
    z_stick = np.zeros(theta.shape[0])
    m = np.hstack((z_stick, -r_hat[0, 2, :], r_hat[0, 1, :], r_hat[0, 2, :], z_stick,
                   -r_hat[0, 0, :], -r_hat[0, 1, :], r_hat[0, 0, :], z_stick)).reshape(
                       (3, 3, -1))
    i_cube = np.broadcast_to(
        np.eye(3).reshape((3, 3, 1)), (3, 3, theta.shape[0]))
    A = np.transpose(r_hat, axes=(2, 1, 0))
    B = np.transpose(r_hat, axes=(2, 0, 1))
    dot = np.matmul(A, B).transpose((1, 2, 0))
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R


def with_zeros(x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


def pack(x):
    depth = x.shape[-1]
    return np.hstack((np.zeros((4, 3, depth)), x.reshape(4, 1, depth)))


def smpl_model(model_path, betas, pose, trans):
    with open(model_path, 'rb') as f:
        params = pickle.load(f)

    J_regressor = params['J_regressor']
    weights = params['weights']
    posedirs = params['posedirs']
    v_template = params['v_template']
    shapedirs = params['shapedirs']
    f = params['f']

    kintree_table = params['kintree_table']
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {
        i: id_to_col[kintree_table[0, i]]
        for i in range(1, kintree_table.shape[1])
    }

    v_shaped = shapedirs.dot(betas) + v_template

    J = J_regressor.dot(v_shaped)

    pose_cube = pose[3:].reshape((-1, 1, 3)).transpose((1, 2, 0))

    R_cube = rodrigues(pose_cube)
    I_cube = np.broadcast_to(
        np.eye(3).reshape((3, 3, 1)), (3, 3, R_cube.shape[2]))
    lrotmin = (R_cube - I_cube).transpose(2, 0, 1).ravel()

    v_posed = v_shaped + posedirs.dot(lrotmin)

    pose = pose.reshape((-1, 3))

    depth = kintree_table.shape[1]

    results = np.zeros((4, 4, depth))

    results[:, :, 0] = with_zeros(
        np.hstack((rodrigues_single(pose[0, :]), J[0, :].reshape((3, 1)))))

    for i in range(1, kintree_table.shape[1]):
        results[:, :, i] = results[:, :, parent[i]].dot(
            with_zeros(
                np.hstack((rodrigues_single(pose[i, :]),
                           ((J[i, :] - J[parent[i], :]).reshape((3, 1)))))))


    results = results - pack(
        np.matmul(
            results.transpose((2, 0, 1)),
            np.hstack(
                (
                    J,
                    np.zeros((24, 1))
                )
            ).reshape((24, 4, 1)).transpose((0, 1, 2))
        ).transpose((1, 2, 0))
    )

    T = results.dot(weights.T)

    rest_shape_h = np.vstack((v_posed.T, np.ones((1, v_posed.shape[0]))))


    v = np.matmul(
        T.transpose((2, 0, 1)),
        rest_shape_h.reshape((4, 1, -1)).transpose((2, 0, 1))).reshape(
            (-1, 4))[:, :3]

    result = v + trans.reshape((1, 3))

    return result, f


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    pose = (np.random.rand(pose_size) - 0.5) * 0.4
    betas = (np.random.rand(beta_size) - 0.5) * 0.06
    trans = np.zeros(3)

    result, faces = smpl_model('./model.pkl', betas, pose, trans)

    outmesh_path = './hello_smpl.obj'
    with open(outmesh_path, 'w') as fp:
        for v in result:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
