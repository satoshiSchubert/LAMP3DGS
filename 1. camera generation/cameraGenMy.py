import numpy as np
import random
import json
from math import *


def getViewMatrix(vp, target, up):
    vp = np.array(vp)
    target = np.array(target)
    up = np.array(up)
    
    front = target - vp
    hori = np.cross(front, up)
    rup = np.cross(hori, front)

    hori = hori/np.linalg.norm(hori)
    front = front/np.linalg.norm(front)
    rup = rup/np.linalg.norm(rup)
    
    ans = np.zeros((4,4))
    ans[0:3, 0] = -hori
    ans[0:3, 1] = rup
    ans[0:3, 2] = front
    ans[0:3,3] = vp
    ans[3,3] = 1.0

    return ans
def getCmrPos():
    cmrs = []

    idx = 0
    
    length = 15
    # 1st rotate camera
    direction = [0.376047, -0.758426, -0.532333]
    for i in range(180):
        r=length
        theta = i * 2 / 180 * pi
        phi = 60 / 180 * pi
        pos = [
            round(r * sin(phi) * cos(theta) +center[0], 5),
            round(r * cos(phi)              +center[1], 5),
            round(r * sin(phi) * sin(theta) +center[2], 5)
        ]
        lookat = [
            round(center[0], 3),
            round(center[1], 3),
            round(center[2], 3)
        ]
        viewMat = getViewMatrix(pos, lookat, np.array([0, 1, 0], dtype=np.float32)).tolist()
        cmrs.append({
            'pos' : pos, 'lookat' : lookat, 'direction' : direction, 'viewMat' : viewMat, 'id' : idx
        })
        idx += 1

    for i in range(180):
        r=length
        theta = 0
        phi = 60 / 180 * pi
        pos = [
            round(r * sin(phi) * cos(theta) +center[0], 5),
            round(r * cos(phi)              +center[1], 5),
            round(r * sin(phi) * sin(theta) +center[2], 5)
        ]
        lookat = [
            round(center[0], 3),
            round(center[1], 3),
            round(center[2], 3)
        ]
        rtheta = i*2/180*pi
        rot = np.array([
            [cos(rtheta), 0, sin(rtheta)],
            [0, 1, 0],
            [-sin(rtheta), 0, cos(rtheta)]
        ])
        ans = rot @ direction
        ans2 = [
            round( ans[0], 5 ),
            round( ans[1], 5 ),
            round( ans[2], 5 )
        ]
        viewMat = getViewMatrix(pos, lookat, np.array([0, 1, 0], dtype=np.float32)).tolist()
        cmrs.append({
            'pos' : pos, 'lookat' : lookat, 'direction' : ans2, 'viewMat' : viewMat, 'id' : idx
        })
        idx += 1
    return cmrs


def getViewMatrix(vp, target, up):
    vp = np.array(vp)
    target = np.array(target)
    up = np.array(up)
    
    front = target - vp
    hori = np.cross(front, up)
    rup = np.cross(hori, front)

    hori = hori/np.linalg.norm(hori)
    front = front/np.linalg.norm(front)
    rup = rup/np.linalg.norm(rup)
    
    ans = np.zeros((4,4))
    ans[0:3, 0] = -hori
    ans[0:3, 1] = rup
    ans[0:3, 2] = front
    ans[0:3,3] = vp
    ans[3,3] = 1.0

    return ans


def randPos(idd):
    length = 15
    
    theta = random.random() * 2 * pi
    if idd < 515:
        phi = random.random() * pi * 5 / 8
    elif idd < 824:
        phi = random.random() * pi * 3 / 8 + pi*5/8
    else:
        phi = random.random() * pi
    
    r = random.uniform(0.6, 1.1) * length
    pos = [
        round(r * sin(phi) * cos(theta) +center[0], 3),
        round(r * cos(phi)              +center[1], 3),
        round(r * sin(phi) * sin(theta) +center[2], 3)
    ]
    lookat = [
        round(random.uniform(-0.3, 0.3)+center[0], 3),
        round(random.uniform(-0.3, 0.3)+center[1], 3),
        round(random.uniform(-0.3, 0.3)+center[2], 3)
    ]

    ltheta = random.random() * 2 * pi
    lphi = random.random() * pi
    direction = [
        round(sin(lphi) * cos(ltheta), 3),
        round(cos(lphi)             , 3),
        round(sin(lphi) * sin(ltheta), 3)
    ]

    viewMat = getViewMatrix(pos, lookat, np.array([0, 1, 0], dtype=np.float32)).tolist()
    
    return {'pos' : pos, 'lookat' : lookat, 'direction' : direction, 'viewMat' : viewMat, 'id' : idd}


def old_jkw_code():
    center = np.array([0.005450010299682617, 12.410860061645508, -0.191869854927063])
    cmr = np.array([-10.667699813842773, 14.31410026550293, 10.2878999710083])
    distance = 15

    ans = getCmrPos()
    temp = json.dumps(ans)
    with open('cfgs-video.json', 'w') as f:
        f.write(temp)

    # video camera with envmap

    with open('cfgs-video.json', 'r') as f:
        ans = json.load(f)

    out = []
    for iter, i in enumerate(ans):
        temp = i
        del temp['direction']
        if iter < 180:
            temp['envRot'] = 0
        else:
            temp['envRot'] = (iter - 180) * 2
        out.append(temp)

    with open('cfgs-envvideo.json', 'w') as f:
        json.dump(out, f)

    # train & test set camera generation

    center = np.array([0.005450010299682617, 12.410860061645508, -0.191869854927063], dtype=np.float32)
    cmr = np.array([-10.667699813842773, 14.31410026550293, 10.2878999710083], dtype=np.float32)
    distance = 15

    np.random.seed(0)
    random.seed(0)


    ans = []
    for i in range(2100):
        ans.append(randPos(i))

    temp = json.dumps(ans)
    with open('cfgs-2100.json', 'w') as f:
        f.write(temp)

    # train & test set camera generation with envmap

    with open('cfgs-2100.json', 'r') as f:
        ans = json.load(f)
    random.seed(0)

    out = []
    for i in ans:
        temp = i
        del temp['direction']
        temp['envRot'] = random.randint(0, 359)
        out.append(temp)

    with open('cfgs-env.json', 'w') as f:
        json.dump(out, f)

import gzip, pickle
def main():
    pickle_path = "../shtdata/light_pickle/"
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    case1_pth = pickle_path+"case1.pkl"
    case2_pth = pickle_path+"case2.pkl"
    case3_pth = pickle_path+"case3.pkl"
    with gzip.open(case1_pth + '.gz', 'rb') as lf:
        localData = pickle.load(lf)
        print(localData.keys())

        # get camera position
        cam_pos = localData['camera_pos'].reshape(-1, 3)

        output = []
        # calculate other mats
        for i in range(cam_pos.shape[0]):
            pos = cam_pos[i]
            lookat = center
            direction = lookat - pos
            viewMat = getViewMatrix(pos, lookat, np.array([0, 1, 0], dtype=np.float32)).tolist()
            #pack and save to json
            output.append({
                'pos' : pos.tolist(), 'lookat' : lookat.tolist(), 'direction' : direction.tolist(), 'viewMat' : viewMat, 'id' : i
            })
        # save to json
        with open('cfgs-sht.json', 'w') as f:
            json.dump(output, f)

    lf.close()

if __name__ == "__main__":
    main()














