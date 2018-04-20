import numpy as np
import xml.etree.ElementTree as ET
import smpl_np


def parse_marker_pp(path):
    klist = []
    plist = []
    root = ET.parse(path).getroot()
    for p in root.findall('point'):
        attrib = p.attrib
        coordinates = np.array([
            float(attrib['x']),
            float(attrib['y']),
            float(attrib['z'])
        ])
        klist.append(attrib['name'])
        plist.append(coordinates)
    points = np.stack(plist, axis=0)
    return klist, points


def marker_on_surface():
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
        'RSHO': 11745
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

