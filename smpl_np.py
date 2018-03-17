import numpy as np
import pickle
import cv2


def rodrigues(r):
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    r_hat = r / (theta + 1e-8)
    cos = np.cos(theta) # theta is radius
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack((z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
                   -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick)).reshape(
                       (-1, 3, 3))
    i_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), (theta.shape[0], 3, 3))
    A = np.transpose(r_hat, axes=(0, 2, 1))
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R


def with_zeros(x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))


def pack(x):
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))


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
    pose_cube = pose.reshape((-1, 1, 3))
    R_cube_big = rodrigues(pose_cube)
    R_cube = R_cube_big[1:]
    I_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), (R_cube.shape[0], 3, 3))
    lrotmin = (R_cube - I_cube).ravel()
    v_posed = v_shaped + posedirs.dot(lrotmin)
    pose = pose.reshape((-1, 3))
    results = np.empty((kintree_table.shape[1], 4, 4))
    results[0, :, :] = with_zeros(np.hstack((R_cube_big[0], J[0, :].reshape((3, 1)))))
    for i in range(1, kintree_table.shape[1]):
        results[i, :, :] = results[parent[i], :, :].dot(with_zeros(np.hstack((R_cube_big[i], ((J[i, :] - J[parent[i], :]).reshape((3, 1)))))))
    results = results - pack(np.matmul(results, np.hstack((J, np.zeros((24, 1)))).reshape((24, 4, 1))))
    T = np.tensordot(weights, results, axes=((1), (0)))
    rest_shape_h = np.hstack((v_posed, np.ones((v_posed.shape[0], 1))))
    v = np.matmul(T, rest_shape_h.reshape((-1, 4, 1))).reshape((-1, 4))[:, :3]
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

    outmesh_path = './smpl_np.obj'
    with open(outmesh_path, 'w') as fp:
        for v in result:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in faces + 1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
