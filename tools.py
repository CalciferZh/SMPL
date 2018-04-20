import numpy as np
import xml.etree.ElementTree as ET
import smpl_np


def compute_plane(verts, faces, fidx):
    v0, v1, v2 = verts[faces[fidx, 0]], verts[faces[fidx, 1]], verts[faces[fidx, 2]]
    return v0, v1, v2


def marker_face_weights(verts, faces, markers, correspondence):
    weights = {}
    for name, point in markers.items():
        v0, v1, v2 = compute_plane(verts, faces, correspondence[name])
        x1, y1, _ = v1 - v0
        x2, y2, _ = v2 - v0
        x3, y3, _ = point - v0
        n = (x1 * y3 - x3 * y1) / (x1 * y2 - x2 * y1)
        m = (x3 - n * x2) / x1
        o = 1 - m - n
        weights[name] = (o, m, n)
    return weights


def parse_marker_pp(path):
    markers = {}
    root = ET.parse(path).getroot()
    for p in root.findall('point'):
        attrib = p.attrib
        coordinates = np.array([
            float(attrib['x']),
            float(attrib['y']),
            float(attrib['z'])
        ])
        markers[attrib['name']] = coordinates
    return markers


def marker_face_corr():
    ret = {
        'LFWT': 6108,
        'LTHI': 5199, 
        'LWRA': 2462,
        'LBWT': 5357,
        'RBWT': 12244,
        'RFRM': 8827,
        'LFRM': 4482,
        'RWRA': 9679,
        'LFIN': 3353,
        'RFIN': 9416,
        'STRN': 5072,
        'LELB': 2425,
        'RELB': 11260,
        'T8': 6865,
        'T10': 11854,
        'LKNE': 1302,
        'RKNE': 8090,
        'LUPA': 1982,
        'LSHN': 5280,
        'RSHN': 8115,
        'CLAV': 5047,
        'RSHO': 11745,
        'RUPA': 11471,
        'LSHO': 4859,
        'C7': 12340,
        'LBHD': 384,
        'RBHD': 7270,
        'LFHD': 95,
        'RFHD': 6982,
        'LANK': 5742,
        'RANK': 12631,
        'LHEE': 5815,
        'RHEE': 12703,
        'RTOE': 12405,
        'LTOE': 5516,
        'LMT5': 5627,
        'RMT5': 12514,
        'RWRB': 9399,
        'LWRB': 2512,
        'RTHI': 7971,
        'RFWT': 9052,
        'RBAC': 7966
    }
    return ret


def generate_planes(verts, faces):
    planes = []
    for f in faces:
        v1, v2, v3 = verts[f[0]], verts[f[1]], verts[f[2]]
        plane = np.stack([v1, v2, v3], axis=0)
        planes.append(plane)
    planes = np.stack(planes, axis=0)
    return planes


def obj_save(path, vertices, faces=None):
    with open(path, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if faces is not None:
            for f in faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def compute_markers(verts, faces, weights, correspondence):
    markers = {}
    for name, weight in weights.items():
        v0, v1, v2 = compute_plane(verts, faces, correspondence[name])
        markers[name] = v0 * weight[0] + v1 * weight[1] + v2 * weight[2]
    return markers


if __name__ == '__main__':
    path = './pose_prior/CMU_Mocap_Markers.pp'
    markers = parse_marker_pp(path)
    verts, faces = smpl_np.smpl_model(
        'model.pkl',
        np.zeros(10),
        np.zeros((24, 3)),
        np.zeros(3)
    )
    correspondence = marker_face_corr()
    weights = marker_face_weights(verts, faces, markers, correspondence)

    np.random.seed(9608)
    pose = (np.random.rand(24, 3) - 0.5) * 0.4
    beta = (np.random.rand(10) - 0.5) * 0.06
    trans = np.zeros(3)
    verts, faces = smpl_np.smpl_model(
        'model.pkl',
        beta,
        pose,
        trans
    )
    markers = compute_markers(verts, faces, weights, correspondence)
    obj_save('markers.obj', markers.values())
    obj_save('pose.obj', verts, faces)
