
import numpy
import numpy as np
import math
import torch
#import open3d as o3d
import copy
def WrapEqualAreaSquare(uv):
    if (uv[0] < 0):
        uv[0] = -uv[0]
        uv[1] = 1 - uv[1]
    elif (uv[0] > 1):
        uv[0] = 2 - uv[0]
        uv[1] = 1 - uv[1]
    if (uv[1] < 0):
        uv[0] = 1 - uv[0]
        uv[1] = -uv[1]
    elif (uv[1] > 1):
        uv[0] = 1 - uv[0]
        uv[1] = 2 - uv[1]
    return uv


def EqualAreaSquareToSphere(p):
    u = 2 * p[0] - 1
    v = 2 * p[1] - 1
    up = math.fabs(u)
    vp = math.fabs(v)

    signedDistance = 1 - (up + vp)
    d = math.fabs(signedDistance)
    r = 1 - d
    if r == 0:
        phi = 1 * math.pi / 4
    else:
        phi = ((vp - up) / r + 1) * math.pi / 4
    z = math.fabs(1 - r * r)
    if (signedDistance < 0):
        z = -z;
    cosPhi = math.fabs(math.cos(phi))
    if (u < 0):
        cosPhi = - cosPhi
    sinPhi = math.fabs(math.sin(phi))
    if (v < 0):
        sinPhi = - sinPhi
    return (cosPhi * r * math.sqrt(2 - r * r), sinPhi * r * math.sqrt(2 - r * r), z)

def generate_camera_udpate(index):
    u = int(index / 16)
    v = index - u * 16
    u = (0.5 + u) / 16
    v = (0.5 + v) / 16
    uv = [u, v]
    uv = WrapEqualAreaSquare(uv)
    uv = list(uv)
    dir = EqualAreaSquareToSphere(uv)
    dir = np.array(list(dir))
    # position = dir * self.radius
    position = dir * 0.8
    forward = -dir
    return position


import gzip, pickle
import zstandard as zstd
def main():
    pickle_path = "../shtdata/light_pickle/"
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    case1_pth = pickle_path+"case1.pkl"
    case2_pth = pickle_path+"case2.pkl"
    case3_pth = pickle_path+"case3.pkl"

    with gzip.open(case2_pth + '.gz', 'rb') as lf:
        localData = pickle.load(lf)

    # with open(casenew_pth, 'rb') as lf:
    #     dctx = zstd.ZstdDecompressor()
    #     localData = pickle.loads(dctx.decompress(lf.read()))
    #     # print_dict(lightData)

        print(localData.keys())
        # get camera position
        cam_pos = localData['camera_pos'].reshape(-1, 3)

        output = []
        # calculate other mats
        for i in range(cam_pos.shape[0]):
            pos = cam_pos[i]
            lookat = center
            direction = [1,0,0] # 方向光方向，和几何重建无关
            viewMat = getViewMatrix(pos, lookat, np.array([0, 1, 0], dtype=np.float32)).tolist()
            #pack and save to json
            output.append({
                'pos' : pos.tolist(), 'lookat' : lookat.tolist(), 'direction' : direction, 'viewMat' : viewMat, 'id' : i
            })
        # save to json
        with open("../3gaussian-ae/furball/cfgs-case.json", 'w') as f:
            json.dump(output, f)

    lf.close()

if __name__ == "__main__":
    generate_camera_udpate(0)

