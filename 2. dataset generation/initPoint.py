import numpy as np
import os
from math import *
from plyfile import PlyData, PlyElement
import gc

ans = None

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


def inverse_sigmoid(x):
    return np.log(x/(1-x))
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

def savePly(data, props, path : str):
    dtype_full = [(attribute, 'f4') for attribute in props]
    data = data.reshape((data.shape[0], -1))
    elements = np.zeros(data.shape[0], dtype=dtype_full)
    for i, prop in enumerate(props):
        elements[prop] = data[:, i]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

savePly(total, dtype_full, 'point_cloud.ply')
