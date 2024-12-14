import numpy as np
import os
from math import *
from plyfile import PlyData, PlyElement
import gc

from tqdm import tqdm

ans = None

def inverse_sigmoid(x):
    return np.log(x/(1-x))
def savePly(data, props, path : str):
    dtype_full = [(attribute, 'f4') for attribute in props]
    data = data.reshape((data.shape[0], -1))
    elements = np.zeros(data.shape[0], dtype=dtype_full)
    for i, prop in enumerate(props):
        elements[prop] = data[:, i]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
def kevinMain():
    if os.path.exists('./models/hair.bin'):
        ans = np.fromfile('./models/hair.bin', dtype=np.float32).reshape((-1,4,4))
        print(ans.shape)
    else:
        with open('./models/hair.m3hair', 'r') as f:
            temp = f.readlines()
            tans = []
            for line in temp:
                if line != '\n':
                    tans.append([float(x) for x in line.split(' ')])
        ans = np.array(tans, dtype=np.float32).reshape((-1,4,4)) #[::256,:,:]
        ans.tofile('./models/hair.bin')

    # 120, 70, 20 rgb
    # 0.47, 0.27, 0.078

    #idx = np.random.choice(ans.shape[0], ans.shape[0]//8, replace=True)
    ans = ans[::3, :, :]

    p1 = (ans[:,0,0:3]+ans[:,1,0:3])*0.5
    p2 = (ans[:,1,0:3]+ans[:,2,0:3])*0.5
    p3 = (ans[:,2,0:3]+ans[:,3,0:3])*0.5
    xyz = np.hstack((p1,p2,p3)).reshape((-1,3))
    print(xyz.shape)

    norms = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    delta1 = ans[:,1,0:3]-ans[:,0,0:3]
    delta2 = ans[:,2,0:3]-ans[:,1,0:3]
    delta3 = ans[:,3,0:3]-ans[:,2,0:3]
    deltaxyz = np.hstack((delta1,delta2,delta3)).reshape((-1,3))

    lens = np.linalg.norm(deltaxyz, axis=-1)

    # SH ~= (RGB-0.5)/C0
    #    ~= [-0.10634723, -0.81532877, -1.49595105]
    shs = np.zeros((xyz.shape[0], 48), dtype=np.float32)
    shs[:,0] = -0.10634723 #0.47
    shs[:,1] = -0.81532877 #0.27
    shs[:,2] = -1.49595105 #0.078

    print(shs.shape)



    ops = np.full((xyz.shape[0], 1), 0.5, dtype=np.float32)
    ops = inverse_sigmoid(ops)

    print(ops.shape)


    scs = (ans[:, 0:3, 3:]).repeat(3, 2).reshape(-1,3)
    scs[:,2] = lens
    scs = np.log(scs)
    print('Max SCS: {}'.format(scs.max()))

    print(scs.shape)


    rot = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    rot[:, 0:3] = np.cross( np.array([0,0,1],dtype=np.float32).reshape(1,3).repeat(xyz.shape[0], 0), deltaxyz )
    rot[:, 3] = 1.0 + deltaxyz[:, 2]
    rot = rot / np.linalg.norm(rot, axis=-1)[:,None]
    #rot[:,3] = 1.0
    print(rot.shape)

    total = np.hstack((xyz, norms, shs, ops, scs, rot)).reshape(xyz.shape[0], -1)

    print(total.shape)

    #del xyz
    #del norms
    #del shs
    #del ops
    #del scs
    #del rot
    #gc.collect()

    #total = total[::4,:]

    print('sample!')

    dtype_full = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'f_rest_0', 'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4', 'f_rest_5', 'f_rest_6', 'f_rest_7', 'f_rest_8', 'f_rest_9', 'f_rest_10', 'f_rest_11', 'f_rest_12', 'f_rest_13', 'f_rest_14', 'f_rest_15', 'f_rest_16', 'f_rest_17', 'f_rest_18', 'f_rest_19', 'f_rest_20', 'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24', 'f_rest_25', 'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29', 'f_rest_30', 'f_rest_31', 'f_rest_32', 'f_rest_33', 'f_rest_34', 'f_rest_35', 'f_rest_36', 'f_rest_37', 'f_rest_38', 'f_rest_39', 'f_rest_40', 'f_rest_41', 'f_rest_42', 'f_rest_43', 'f_rest_44', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    print(total.dtype)
    print(total[0])
    savePly(total, dtype_full, 'point_cloud.ply')

import json
import gzip
import pickle
import cv2 as cv
def main():
    pickle_path = "../shtdata/light_pickle/"
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    case1_pth = pickle_path+"case1.pkl"
    case2_pth = pickle_path+"case2.pkl"
    case3_pth = pickle_path+"case3.pkl"
    with gzip.open(case1_pth + '.gz', 'rb') as lf:
        localData = pickle.load(lf)
        print(localData.keys())

        # Initialize Gaussian
        # get screen space world position, and downsampling to 64*64
        world_pos = localData['position'].reshape(-1, 512, 512, 3)
        world_pos = cv.resize(world_pos, (64, 64), interpolation=cv.INTER_LINEAR)
        print(world_pos.shape)
        # 维护一个Gaussian List，对于每一张图片，计算出其中的初始化Gaussian并批量append进list中
        l = 5
        n = 10
        gaussians = []
        for i in range(world_pos.shape[0]):
            pos = world_pos[i]

            # 接下来帮我完成这个算法
            pass

import numpy as np
import cv2 as cv

def sample_gaussians(world_pos, l, n):
    """
    Perform DFS-based sampling to generate Gaussians from non-zero points in the world position map.
    # 假设pos中背景是0，灯具所占的pixel值为其在世界坐标中的位置
            # 设计一个DFS算法，完成如下事情
            # 1. 随机采样其中一个非0点，并采样其周围像素与其距离为l像素的点，若存在这种点，采样其中一个，这两个点即构成一个Gaussian
            # 2. 两个点确定一个gaussian,其scale为l, opacity为1.0, rotation为0,0,0,1，中心为两点world position的中点
            # 3. 根据kevinMain代码中的操作，组成gaussian，加入到list中，且将这两个点的像素值设为0，表示已经被采样
            # 4. 重复1-3，直到list中的gaussian数量达到n，或不存在满足条件的点
            # 5. 结束这一frame的采样，对下一frame重复1-4

    Args:
        world_pos (np.ndarray): A single frame of world position data, shape (64, 64, 3).
        l (float): Distance threshold for sampling.
        n (int): Maximum number of Gaussians to sample per frame.

    Returns:
        list: List of sampled Gaussians. Each Gaussian is a dictionary containing keys:
              - 'center': Gaussian center (3D world position).
              - 'scale': Scale of the Gaussian (3-channel scale).
              - 'opacity': Opacity value.
              - 'rotation': Rotation quaternion.
    """
    # Copy input to avoid modifying the original data
    frame = world_pos.copy()
    gaussians = []

    # Helper function to find neighbors within pixel distance l
    def find_neighbor(pos_idx, visited):
        x1, y1 = pos_idx
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if tuple((i, j)) in visited:
                    continue
                if np.sqrt((i - x1) ** 2 + (j - y1) ** 2) <= l and np.any(frame[i, j] != 0):
                    return (i, j)
        return None

    visited = set()

    while len(gaussians) < n:
        # Find a random non-zero point
        non_zero_indices = np.argwhere(np.any(frame != 0, axis=-1))
        if len(non_zero_indices) == 0:
            break

        idx = tuple(non_zero_indices[np.random.choice(len(non_zero_indices))])
        if idx in visited:
            continue

        # Mark this point as visited
        visited.add(idx)
        pos1 = frame[idx]

        # Find a neighbor within pixel distance l
        neighbor_idx = find_neighbor(idx, visited)
        if neighbor_idx is None:
            continue

        visited.add(neighbor_idx)
        pos2 = frame[neighbor_idx]

        # Compute Gaussian parameters
        xyz = (pos1 + pos2) / 2
        scale = np.array([0.2 * l, 0.2 * l, l], dtype=np.float32)  # 3-channel scale

        shs = np.zeros((48,), dtype=np.float32)
        shs[0] = -0.10634723  # 0.47
        shs[1] = -0.81532877  # 0.27
        shs[2] = -1.49595105  # 0.078

        opacity = np.full((1,), 0.5, dtype=np.float32)
        opacity = inverse_sigmoid(opacity)
        norms = np.zeros((3,), dtype=np.float32)

        deltaxyz = pos2 - pos1
        rot = np.zeros((4,), dtype=np.float32)
        rot[0:3] = np.cross(np.array([0, 0, 1], dtype=np.float32).reshape(1, 3), deltaxyz)
        rot[3] = 1.0 + deltaxyz[2]
        rot = rot / np.linalg.norm(rot, axis=-1)

        total = np.hstack((xyz, norms, shs, opacity, scale, rot))

        # Add Gaussian to the list
        gaussians.append(total)

        # Mark sampled points as zero
        frame[idx] = 0
        frame[neighbor_idx] = 0

    # 把list中的gaussian按照第一维度转成一个numpy array
    gaussians = np.array(gaussians, dtype=np.float32)
    return gaussians

def main_sampling():
    pickle_path = "../shtdata/light_pickle/"
    case1_pth = pickle_path + "case1.pkl"

    with gzip.open(case1_pth + '.gz', 'rb') as lf:
        localData = pickle.load(lf)
        world_pos = localData['position'].reshape(-1, 512, 512, 3)
        world_pos = np.array([cv.resize(frame, (64, 64), interpolation=cv.INTER_LINEAR) for frame in world_pos])

        print(f"Resized world_pos shape: {world_pos.shape}")

        # L和N与resize后的分辨率有关！因为实质是利用下采样导致的稀疏化来采样Gaussian
        l = 5 # Distance threshold for two sampled points
        n = 10 # Maximum number of Gaussians to sample per frame
        all_gaussians = []
        print("total frames:", len(world_pos))
        for i, frame in tqdm(enumerate(world_pos)):
            gaussians = sample_gaussians(frame, l, n)
            all_gaussians.append(gaussians)
            print(f"Frame {i}: {len(gaussians)} Gaussians sampled.")
        lf.close()
        all_gaussians = np.array(all_gaussians, dtype=np.float32)
        all_gaussians = all_gaussians.reshape((-1, all_gaussians.shape[-1]))
        #return all_gaussians
        dtype_full = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'f_rest_0', 'f_rest_1', 'f_rest_2',
                      'f_rest_3', 'f_rest_4', 'f_rest_5', 'f_rest_6', 'f_rest_7', 'f_rest_8', 'f_rest_9', 'f_rest_10',
                      'f_rest_11', 'f_rest_12', 'f_rest_13', 'f_rest_14', 'f_rest_15', 'f_rest_16', 'f_rest_17',
                      'f_rest_18', 'f_rest_19', 'f_rest_20', 'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24',
                      'f_rest_25', 'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29', 'f_rest_30', 'f_rest_31',
                      'f_rest_32', 'f_rest_33', 'f_rest_34', 'f_rest_35', 'f_rest_36', 'f_rest_37', 'f_rest_38',
                      'f_rest_39', 'f_rest_40', 'f_rest_41', 'f_rest_42', 'f_rest_43', 'f_rest_44', 'opacity',
                      'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

        print(all_gaussians[0])
        savePly(all_gaussians, dtype_full, 'point_cloud_my.ply')

if __name__ == "__main__":
    main_sampling()