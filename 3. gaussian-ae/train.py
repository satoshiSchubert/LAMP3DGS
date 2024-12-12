import random
import numpy as np
import time
import torch
t = torch

from tqdm import tqdm

import os
import pickle
import argparse

from model.gaussian_model import GaussianModel
from gaussian_renderer import render_multichannel
from utils.loss_utils import * # ssim/l1, l2
from utils.image_utils import * # mse/psnr
from utils.preprocess import * # load/save exr
from evaluate import generateComparePath, evaluate

from model.pixel_generator import PixelGenerator

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


gpuid = None
tdev=None
trainVersion = None


class Param:
    @staticmethod
    def mergeParam(p0, *ls):
        fs = p0.param
        for i in ls:
            fs.update(i.param)
        return Param(fs)

    def __init__(self, initLs={}):
        self.param = initLs
    
    def __getitem__(self, key):
        if key in self.param:
            return self.param[key]
        else:
            return None
    
    def __setitem__(self, key, value):
        self.param[key] = value

    def __delitem__(self, name: str) -> None:
        if name in self.param:
            del self.param[name]

    def __getattr__(self, name: str):
        if name in self.param:
            return self.param[name]
        else:
            return None

    def tryGet(self, key, default = None):
        if key in self.param:
            return self.param[key]
        else:
            return default
        
    def update(self, ls):
        self.param.update(ls)

pipeParam = Param({
    'convert_SHs_python' : False,
    'compute_cov3D_python' : False,
    'debug' : False
})
optParam = Param({
    'position_lr_init' : 0.00016,
    'position_lr_final' : 0.0000016,
    'position_lr_delay_mult' : 0.01,
    'position_lr_max_steps' : 30_000, #60_000, #30_000,
    'feature_lr' : 0.0025,
    'extra_lr' : 0.0025,
    'opacity_lr' : 0.05,
    'scaling_lr' : 0.005,
    'rotation_lr' : 0.001,
    'percent_dense' : 0.0006, #0.001 #0.01,
    'lambda_dssim' : 0.2,
    'densification_interval' : 100, #100,
    'opacity_reset_interval' : 3000,
    'densify_from_iter' : 1_000, #500,
    'densify_until_iter' : 15_000, #15_000,
    'opacity_thres' : 0.005, # 0.008 #0.05
    'densify_grad_threshold' : 0.00016, # 0.00018 # 0.0002
    'random_background' : False
})
trainParam = Param({
    'bg_color' : torch.tensor([0,0,0], dtype=torch.float32, device=tdev),
    'firstIter' : 1,
    'maxIter' : 60_000, #60_000, #30_000,
    'savePer' : 30_000,
    'trackPer' : 10_000,
    'trackId' : 0,
    'checkpoint' : None, #'./furball/pc/{}_checkpoint_30000.pth'.format(trainVersion),
    'initFile' : './furball/furball.ply', #'./furball/pc/{}_pc_30000.ply'.format(trainVersion),
    'batchSize' : 1
})
runtimeParam = Param({
    'pipeline' : 'relighting',
    'geoloss' : True,
    'geolossUntil' : 100_000,

    'cmrPath' : './furball/cfgs.json',
    'dsName' : 'gt-2100',
    'dsType' : 'directional', # env / enva / directional
    'trainSize' : 2000,
    'testSize' : 100,
    'tag' : '241127_test1',
    'device' : 0,
    'seed' : 0
})

totalParam = Param.mergeParam(pipeParam, optParam, trainParam, runtimeParam)

class Camera:
    def __init__(self, width, height, fovx, fovy, pos, vt, pt, ldir, dist, id):
        # notice lightDir might be polymorphic
        self.width = width
        self.height = height
        self.fovx=fovx
        self.fovy=fovy
        self.camera_center=pos
        self.world_view_transform=vt
        self.full_proj_transform=pt
        self.lightDir = ldir
        self.dist = dist
        self.id = id

def getPixelCamera(w, h, fx, fy):
    ans = None
    i, j = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32), indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    K = np.array([fx, 0, w/2, 0, fy, h/2, 0, 0, 1], dtype=np.float32).reshape((3,3))
    ans = np.dot(xy1, np.linalg.inv(K).T)
    ans = torch.tensor(ans)
    return ans
pixelCamera = None
bgImg = None

def readEnvCfgs(cfgPath):
    print('Reading Cameras from {}'.format(cfgPath))
    with open(cfgPath) as f:
        cmrs = json.load(f)
    data = {}
    params = ['pos', 'lookat', 'envRot', 'id']
    for param in params:
        temp = []
        for cfg in cmrs:
            temp.append(cfg[param])
        data[param] = np.array(temp, dtype=np.float32)
    return data

def readDirectionalCfgs(cfgPath):
    # read cfg
    print('Loading Cameras from {}'.format(cfgPath))
    with open(cfgPath) as f:
        cmrs = json.load(f)

    params = ['pos', 'lookat', 'direction', 'id']
    data = {}
    for param in params:
        temp = []
        for cfg in cmrs:
            temp.append(cfg[param])
        data[param] = np.array(temp, dtype=np.float32)
    return data

def readCommonCfgs(cfgPath):
    # read cfg
    print('Loading Cameras from {}'.format(cfgPath))
    with open(cfgPath) as f:
        cmrs = json.load(f)

    params = ['pos', 'lookat', 'var', 'id']
    data = {}
    for param in params:
        temp = []
        for cfg in cmrs:
            temp.append(cfg[param])
        data[param] = np.array(temp, dtype=np.float32)
    return data

def readAllConfig(kargs : Param):
    cfgPath = kargs.tryGet('cmrPath', './furball/cfgs.json')

    if kargs.dsType == 'directional':
        cmrPacket = readDirectionalCfgs(cfgPath)
        direction = cmrPacket['direction']
    elif kargs.dsType == 'env' or kargs.dsType == 'enva':
        cmrPacket = readEnvCfgs(cfgPath)
        direction = cmrPacket['envRot'] / 360
    else:
        cmrPacket = readCommonCfgs(cfgPath)
        direction = cmrPacket['var']

    pos, lookat, ids = cmrPacket['pos'], cmrPacket['lookat'], cmrPacket['id']

    def getViewMatrix(pos, lookat):
        up = np.array([[0, 1.0, 0]], dtype=np.float32).repeat(pos.shape[0], axis=0)
        front = lookat - pos
        hori = np.cross(front, up)
        rup = np.cross(hori, front)
        front = front / np.linalg.norm(front, axis=1).reshape(-1, 1)
        hori = hori / np.linalg.norm(hori, axis=1).reshape(-1, 1)
        rup = rup / np.linalg.norm(rup, axis=1).reshape(-1, 1)
        viewMat = np.zeros((pos.shape[0], 4, 4), dtype=np.float32)
        viewMat[:, 0, 0:3] = hori
        viewMat[:, 1, 0:3] = -rup
        viewMat[:, 2, 0:3] = front
        viewMat = viewMat.transpose((0,2,1))
        # now as json
        viewMat[:, 0:3, 3] = pos
        viewMat[:, 3, 3] = 1.0
        viewMat = np.linalg.inv(viewMat)
        return viewMat
    viewMat = getViewMatrix(pos, lookat)
        
    # camera const property
    if kargs.cmrProperty is not None:
        cmrProperty = kargs.cmrProperty
    else:
        cmrProperty = {
            'width' : 512,
            'height' : 512,
            #'fy' : 812,
            #'fx' : 812,
            'fovx' : 35,
            'fovy' : 35,
            #'tanx' : 0.3153,
            #'tany' : 0.3153,
            'znear' : 0.01,
            'zfar' : 1000,
            'zsign' : 1.0
        }
    tx = np.tan(cmrProperty['fovx']/360*np.pi)
    ty = np.tan(cmrProperty['fovy']/360*np.pi)
    cmrProperty['tanx'] = tx
    cmrProperty['tany'] = ty
    fx = cmrProperty['width']/tx*0.5
    fy = cmrProperty['height']/ty*0.5
    cmrProperty['fx'] = fx
    cmrProperty['fy'] = fy
    print('Cmr: tan {}/{}, f {}/{}'.format(tx, ty, fx, fy))

    def getProjectionMatrix(cfg : dict) -> np.array:
        znear = cfg['znear']
        zfar = cfg['zfar']
        zsign = cfg['zsign']
        tanx = cfg['tanx']
        tany = cfg['tany']
        top = tany * znear
        bottom = -top
        right = tanx * znear
        left = -right
        mat = np.zeros((4,4), dtype=np.float32)
        mat[0, 0] = 2.0 * znear / (right - left)
        mat[1, 1] = 2.0 * znear / (top - bottom)
        mat[2, 0] = (right + left) / (right - left)
        mat[2, 1] = (top + bottom) / (top - bottom)
        mat[2, 3] = zsign
        mat[2, 2] = zsign * zfar / (zfar - znear)
        mat[3, 2] = -(zfar * znear) / (zfar - znear)
        return mat
    proj = getProjectionMatrix(cmrProperty)

    global pixelCamera
    global tdev
    assert tdev is not None

    if pixelCamera is None:
        pixelCamera = getPixelCamera(cmrProperty['width'], cmrProperty['height'], cmrProperty['fx'], cmrProperty['fy']).to(tdev)
    
    tPos = torch.from_numpy(pos).to(tdev)
    tView = torch.from_numpy(viewMat).transpose(1,2).contiguous().to(tdev)
    tProj = torch.from_numpy(proj).to(tdev)
    tProj.unsqueeze_(0)
    tProj = tProj.repeat(tPos.shape[0], 1, 1)
    tProj = tView @ tProj

    dist = lookat - pos
    dist = np.linalg.norm(dist, axis=1)

    cmr = []
    for i in range(pos.shape[0]):
        cmr.append(
            Camera(
                cmrProperty['width'], cmrProperty['height'],
                cmrProperty['fovx'] / 180 * np.pi, cmrProperty['fovy'] / 180 * np.pi,
                tPos[i], tView[i], tProj[i], direction[i], dist[i], ids[i]
            )
        )

    return cmr

datasetChannels = [
    'R', 'G', 'B', 'A', # 0~4
    'alb.R', 'alb.G', 'alb.B', #4~7
    'nn.X', 'nn.Y', 'nn.Z', #7~10
    'pos.X', 'pos.Y', 'pos.Z', #10~13
    'dp.T'
]

def tryLoadCache(path, cb, *args, **kargs):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            tempList = pickle.load(f)
        return tempList
    else:
        tempList = cb(*args, **kargs)
        if tempList is not None and (
            type(tempList) == list and len(tempList)> 0
        ):
            with open(path, 'wb') as f:
                pickle.dump(tempList, f)
        return tempList

def readDirectionalDataset(kargs : Param):
    def loadExrs(pathList, channels):
        ans = []
        cnt = len(pathList)
        bar = tqdm(range(1, cnt+1), desc='LoadPic', dynamic_ncols=True)
        for elem in pathList:
            i, path = elem
            ans.append((i, readExr(path, channels) ))
            bar.update(1)
        bar.close()
        return ans
    cnt = kargs.trainSize + kargs.testSize
    dsName = kargs.tryGet('dsName', 'gt')
    rawBase = './furball/{}/furball-{}.exr'
    denoisedBase = './furball/{}/furball-{}-denoised.exr'
    rawPath = ([(i, rawBase.format(dsName, i)) for i in range(0, cnt)])
    denoisedPath = ([(i, denoisedBase.format(dsName, i)) for i in range(0, cnt)])
    rawFile = tryLoadCache(
        './furball/binfile/{}-rawPic.bin'.format(dsName),
        loadExrs, rawPath, datasetChannels[10:13])
    denoisedFile = tryLoadCache(
        './furball/binfile/{}-denoisedPic.bin'.format(dsName),
        loadExrs, denoisedPath, datasetChannels[0:4])
    rshape = rawFile[0][1].shape
    dshape = denoisedFile[0][1].shape
    print('Raw Files: {} * {}x{}x{}'.format(len(rawFile), rshape[0], rshape[1], rshape[2]))
    print('Denoised Files: {} * {}x{}x{}'.format(len(denoisedFile), dshape[0], dshape[1], dshape[2]))
    return rawFile, denoisedFile

def readEnvDataset(kargs):
    def loadEnvExrs(rawPath, denoisedPath, bgPath):
        ans = []
        pack = list(zip(rawPath, denoisedPath, bgPath))
        cnt = len(pack)
        bar = tqdm(range(1, cnt+1), desc='LoadPic', dynamic_ncols=True)
        for r, d, b in pack:
            assert r[0] == d[0] and r[0] == b[0]
            img = readExr(d[1], datasetChannels[0:3])
            gb = readExr(r[1], datasetChannels[10:13])
            bg = readExr(b[1], datasetChannels[0:3])
            ans.append((d[0], torch.cat([img, gb, bg]) ))
            bar.update(1)
        bar.close()
        return ans
    
    dsName = kargs.tryGet('dsName', 'gt-env')
    rawBase = './furball/{}/furball-{}.exr'
    denoisedBase = './furball/{}/furball-{}-denoised.exr'
    bgBase = './furball/{}/furballbg-{}.exr'

    cnt = kargs.trainSize + kargs.testSize

    rawPath = [(i, rawBase.format(dsName, i)) for i in range(0, cnt)]
    denoisedPath = [(i, denoisedBase.format(dsName, i)) for i in range(0, cnt)]
    bgPath = [(i, bgBase.format(dsName, i)) for i in range(0, cnt)]

    gts = tryLoadCache(
        './furball/binfile/{}-data.bin'.format(dsName),
        loadEnvExrs,
        rawPath, denoisedPath, bgPath
    )
    
    rshape = gts[0][1].shape
    print('Raw Files: {} * {}x{}x{}'.format(len(gts), rshape[0], rshape[1], rshape[2]))
    return gts

def readEnvaDataset(kargs):
    def loadEnvaExrs(rawPath, denoisedPath):
        ans = []
        pack = list(zip(rawPath, denoisedPath))
        cnt = len(pack)
        bar = tqdm(range(1, cnt+1), desc='LoadPic', dynamic_ncols=True)
        for r, d, b in pack:
            assert r[0] == d[0] and r[0] == b[0]
            img = readExr(d[1], datasetChannels[0:4])
            gb = readExr(r[1], datasetChannels[10:13])
            ans.append((d[0], torch.cat([img, gb]) ))
            bar.update(1)
        bar.close()
        return ans
    
    dsName = kargs.tryGet('dsName', 'gt-enva')
    rawBase = './furball/{}/furball-{}.exr'
    denoisedBase = './furball/{}/furball-{}-denoised.exr'

    cnt = kargs.trainSize + kargs.testSize

    rawPath = [(i, rawBase.format(dsName, i)) for i in range(0, cnt)]
    denoisedPath = [(i, denoisedBase.format(dsName, i)) for i in range(0, cnt)]

    gts = tryLoadCache(
        './furball/binfile/{}-data.bin'.format(dsName),
        loadEnvaExrs,
        rawPath, denoisedPath
    )
    
    rshape = gts[0][1].shape
    print('Raw Files: {} * {}x{}x{}'.format(len(gts), rshape[0], rshape[1], rshape[2]))
    return gts

def readDataset(kargs):
    if kargs.dsType == 'env':
        return readEnvDataset(kargs)
    elif kargs.dsType == 'enva':
        return readEnvaDataset(kargs)
    else:
        return readDirectionalDataset(kargs)


def training_report(tb_writer, iteration, loss, component, testing_iterations, scene, renderFunc, renderArgs):
    if tb_writer:
        #tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration); l1_loss
        for name, var in component.items():
            tb_writer.add_scalar('train_loss_patches/{}'.format(name), var.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('total_points', scene.gaussian.get_xyz.shape[0], iteration)
    return


# params:
#   bg_color = [0, 0, 0]
#   firstIter = 0
#   maxIter = 30_000

class Scene:
    def __init__(
        self, gaussian, params,
        camera, datasets
    ):
        self.gaussian = gaussian
        self.params = params
        self.camera = camera
        
        if datasets is not None:
            trainSize = params.trainSize
            testSize = params.testSize

            if params.dsType == 'directional':
                self.datasets = datasets[0] # raw
                self.denoisedSets = datasets[1] # denoised
                self.trainSet = self.denoisedSets[0:trainSize]
                self.testSet = self.denoisedSets[trainSize:trainSize+testSize]
            elif params.dsType == 'env' or params.dsType == 'enva':
                self.datasets = datasets
                self.trainSet = self.datasets[0:trainSize]
                self.testSet = self.datasets[trainSize:trainSize+testSize]
            print('Dataset Length: {} ({}+{})'.format(len(self.trainSet)+len(self.testSet), len(self.trainSet), len(self.testSet)))
        else:
            print('Warning: Empty Datset!')

        gsOutChannel = 33 # senrgb, feature, pos

        if params.dsType == 'env' or params.dsType == 'enva':
            varChannel = 3+1
        else:
            varChannel = 3+3 # rayDir + lightDir
        outChannel = 4
        self.aeModel = PixelGenerator(gsOutChannel, varChannel, outChannel, 512, 8).to(tdev)
        self.geoModel = None
        if params.geoloss:
            geoChannel = 32
            #geoChannel = 16 # divide
            geoChannel = 1 # no mlp for now
            geoOutChannel = 6
            geoVarChannel = 0 # rayDir
            self.geoModel = PixelGenerator(geoChannel, geoVarChannel, geoOutChannel, 1, 1).to(tdev)
        self.radius = 0
        for cmr in self.camera:
            if cmr.dist > self.radius:
                self.radius = cmr.dist
        self.radius *= 1.1

    def load(self, path):
        self.gaussian.loadPly(path)
    def save(self, path):
        self.gaussian.savePly(path)

    def setup(self, kargs : Param):
        checkpoint = self.params['checkpoint']
        if checkpoint:
            if kargs.geoloss:
                temp = torch.load(checkpoint, map_location=tdev)
                (modelParam, aeModel, geoModel, firstIter) = temp
                self.geoModel.load_state_dict(geoModel)
            else:
                (modelParam, aeModel, firstIter) = torch.load(checkpoint, map_location=tdev)
            self.aeModel.load_state_dict(aeModel)
            self.gaussian.restore(modelParam, self.params, self.aeModel, self.geoModel)
            self.firstIter = firstIter
        else:
            self.load(kargs.initFile)
            self.firstIter = kargs.tryGet('firstIter', 1)
        self.gaussian.training_setup(kargs, self.aeModel, self.geoModel)


    def render(self, id, kargs, device=tdev, relighting=True, **others):
        res = (self.camera[id].width, self.camera[id].height)
        global bgImg
        if bgImg is None:
            bgImg = torch.zeros((4+32+3, res[1], res[0]), device=tdev)

        # 3d gs
        rpkg = render_multichannel(self.camera[id], self.gaussian, kargs, bgImg, device=tdev)
        img, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
        
        if kargs.pipeline == 'legacy' or not relighting:
            imgOutput = img.permute(1, 2, 0)
            return rpkg, imgOutput

        # light information
        lightDir = torch.tensor(self.camera[id].lightDir, dtype=t.float32, device=tdev).reshape(-1,1,1).repeat(1, img.shape[1], img.shape[2])
        # pe
        #lightDir = torch.zeros((4,3,2), dtype=t.float32)
        #freq = [np.pi * 2**0, np.pi * 2**1, np.pi * 2**2, np.pi * 2**3]
        #for i in range(4):
        #    for j in range(3):
        #        lightDir[i, j, 0] = np.sin(freq[i] * self.camera[id].lightDir[j])
        #        lightDir[i, j, 1] = np.cos(freq[i] * self.camera[id].lightDir[j])
        #lightDir = lightDir.to(tdev).reshape(-1, 1, 1).repeat(1, img.shape[1], img.shape[2])

        global pixelCamera
        rays_d = (
            self.camera[id].world_view_transform[0:3,0:3] @ 
            pixelCamera.reshape(-1,3).T
        ).T.reshape(res[1], res[0], 3).permute(2, 0, 1)

        itemp = torch.vstack((img[0:36,:,:], rays_d, lightDir))
        itemp.unsqueeze_(0)
        imgInput = itemp.permute(0, 2, 3, 1)

        # no rgb
        imgInput = imgInput[:,:,:,3:36]
        
        # ae
        imgOutput = self.aeModel(imgInput)
        imgOutput.squeeze_(0)
        if kargs.dsType == 'env': # need to concat alpha behind rgb
            imgOutput = imgOutput.permute(2, 0, 1)
            imgOutput = t.cat((imgOutput, img[3:4]))
            imgOutput = imgOutput.permute(1, 2, 0)
        return rpkg, imgOutput
    
    def renderWithGeo(self, id, kargs, device=tdev, **others):
        res = (self.camera[id].width, self.camera[id].height)
        global bgImg
        if bgImg is None:
            bgImg = torch.zeros((4+32+3, res[1], res[0]), device=tdev)

        rpkg = render_multichannel(self.camera[id], self.gaussian, kargs, bgImg, device=tdev)
        img, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
        # light information
        lightDir = torch.tensor(self.camera[id].lightDir, dtype=t.float32, device=tdev).reshape(-1,1,1).repeat(1, img.shape[1], img.shape[2])
        # pe
        #lightDir = torch.zeros((4,3,2), dtype=t.float32)
        #freq = [np.pi * 2**0, np.pi * 2**1, np.pi * 2**2, np.pi * 2**3]
        #for i in range(4):
        #    for j in range(3):
        #        lightDir[i, j, 0] = np.sin(freq[i] * self.camera[id].lightDir[j])
        #        lightDir[i, j, 1] = np.cos(freq[i] * self.camera[id].lightDir[j])
        #lightDir = lightDir.to(tdev).reshape(-1, 1, 1).repeat(1, img.shape[1], img.shape[2])
        
        global pixelCamera
        rays_d = (
            self.camera[id].world_view_transform[0:3,0:3] @ 
            pixelCamera.reshape(-1,3).T
        ).T.reshape(res[1], res[0], 3).permute(2, 0, 1)

        # senrgb
        itemp = torch.vstack((img[3:36,:,:], rays_d, lightDir))
        #itemp = torch.vstack((img[0:36,:,:], rays_d, lightDir))
        itemp.unsqueeze_(0)
        imgInput = itemp.permute(0, 2, 3, 1)

        
        # ae
        imgOutput = self.aeModel(imgInput)
        imgOutput.squeeze_(0)
        
        # geo
        geoOutput = None
        if self.params.geoloss:
            geoOutput = self.geoModel(imgInput[:, :, :, 4:36])
            geoOutput.squeeze_(0)

        return rpkg, imgOutput, geoOutput

    def renderWithGeoDivide(self, id, kargs, device=tdev, **others):
        res = (self.camera[id].width, self.camera[id].height)
        global bgImg
        if bgImg is None:
            bgImg = torch.zeros((4+32+3, res[1], res[0]), device=tdev)

        rpkg = render_multichannel(self.camera[id], self.gaussian, kargs, bgImg, device=tdev)
        img, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
        # light information
        lightDir = torch.tensor(self.camera[id].lightDir, dtype=t.float32, device=tdev).reshape(-1,1,1).repeat(1, img.shape[1], img.shape[2])
        global pixelCamera
        rays_d = (
            self.camera[id].world_view_transform[0:3,0:3] @ 
            pixelCamera.reshape(-1,3).T
        ).T.reshape(res[1], res[0], 3).permute(2, 0, 1)

        # senrgb
        itemp = torch.vstack((img[3:20,:,:], rays_d, lightDir))
        #itemp = torch.vstack((img[0:20,:,:], rays_d, lightDir))
        itemp.unsqueeze_(0)
        imgInput = itemp.permute(0, 2, 3, 1)
        
        # ae
        imgOutput = self.aeModel(imgInput)
        imgOutput.squeeze_(0)
        
        # geo
        geoOutput = None
        if self.params.geoloss:
            geoOutput = self.geoModel(img.permute(1,2,0)[:, :, 20:36].unsqueeze(0))
            geoOutput.squeeze_(0)

        return rpkg, imgOutput, geoOutput
    
    def renderWithExplicitGeo(self, id, kargs, device=tdev, **others):
        res = (self.camera[id].width, self.camera[id].height)
        global bgImg
        if bgImg is None:
            bgImg = torch.zeros((4+32+3, res[1], res[0]), device=tdev)

        rpkg = render_multichannel(self.camera[id], self.gaussian, kargs, bgImg, device=tdev)
        img, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
        # light information
        lightDir = torch.tensor(self.camera[id].lightDir, dtype=t.float32, device=tdev).reshape(-1,1,1).repeat(1, img.shape[1], img.shape[2])
        
        global pixelCamera
        rays_d = (
            self.camera[id].world_view_transform[0:3,0:3] @ 
            pixelCamera.reshape(-1,3).T
        ).T.reshape(res[1], res[0], 3).permute(2, 0, 1)

        # senrgb
        itemp = torch.vstack((img[3:36,:,:], rays_d, lightDir))
        itemp.unsqueeze_(0)
        imgInput = itemp.permute(0, 2, 3, 1)
        
        # ae
        imgOutput = self.aeModel(imgInput)
        imgOutput.squeeze_(0)
        #imgOutput = imgOutput.permute(2, 0, 1)
        #imgOutput = t.cat((imgOutput, img[3:4]))
        #imgOutput = imgOutput.permute(1, 2, 0)
        # geo
        geoOutput = img[36:39, :, :].permute(1,2,0)

        return rpkg, imgOutput, geoOutput

    def mixEnvBG(self, img, bg):
        # img not transposed; bg is transposed
        if bg.device != img.device:
            bg = bg.to(img.device)
        imgtemp = img.permute(2, 0, 1)
        alpha = imgtemp[3]
        return (imgtemp[0:3] * alpha + bg * (1-alpha)).permute(1, 2, 0)

    def trainLossRelighting(self, imgOut, rawGT):
        kargs = self.params
        outTemp = imgOut.unsqueeze(0).permute(0, 3, 1, 2)
        gt = rawGT[0:4,:,:].to(tdev).unsqueeze(0)
        Ll1 = l1_loss(outTemp.unsqueeze(0), gt)
        ssiml = ssim(outTemp, gt)
        loss = (1.0 - kargs.lambda_dssim) * Ll1 + kargs.lambda_dssim * (1.0 - ssiml)
        return loss, { 'Ll1' : Ll1, 'SSIM' : ssiml }
    
    def trainLossRelightingGeo(self, imgOut, geoOut, rawGT):
        kargs = self.params
        outTemp = imgOut.unsqueeze(0).permute(0, 3, 1, 2)
        geoOutTemp = geoOut.unsqueeze(0).permute(0, 3, 1, 2)
        gt = rawGT.to(tdev).unsqueeze(0)
        colorChann = outTemp.shape[1]
        Ll1 = l1_loss(outTemp, gt[:, 0:colorChann])
        ssiml = ssim(outTemp, gt[:, 0:colorChann])
        loss = (1.0 - kargs.lambda_dssim) * Ll1 + kargs.lambda_dssim * (1.0 - ssiml)
        loss2 = l1_loss(geoOutTemp, gt[:,-3:])
        ansLoss = loss * 0.8 + loss2 * 0.2
        return ansLoss, { 'Ll1' : Ll1, 'SSIM' : ssiml, 'Geoloss' : loss2}
        
    def trainLossLegacy(self, imgOut, rawGT):
        kargs = self.params
        outTemp = imgOut.unsqueeze(0).permute(0, 3, 1, 2)[:, 0:3, :, :]
        gt = rawGT[0:3,:,:].to(tdev).unsqueeze(0)
        Ll1 = l1_loss(outTemp.unsqueeze(0), gt)
        loss = (1.0 - kargs.lambda_dssim) * Ll1 + kargs.lambda_dssim * (1.0 - ssim(outTemp, gt))
        return loss


    def startTraining(self, kargs):
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter('./tb/'+totalParam.tag)
        else:
            print("Tensorboard not available: not logging progress")

        firstIter = self.firstIter
        maxIter = self.params.maxIter
        savePer = self.params.savePer

        progress_bar = tqdm(range(firstIter, maxIter + 1), desc='Progress', dynamic_ncols=True)
        firstIter += 1
        
        trainStack = None
        ema_loss_for_log = 0.0

        assert kargs.batchSize == 1
        
        for iter in range(firstIter, maxIter + 1):
            self.gaussian.update_learning_rate(iter)
            if iter % 1000 == 0:
                self.gaussian.oneupSHdegree()

            if not trainStack or len(trainStack) == 0:
                trainStack = self.trainSet.copy()
            randItem = random.randint(0, len(trainStack)-1)
            id, groundTruth = trainStack.pop(randItem)
            
            tempImg, vpt, vf, radii = (None,None,None,None)
            if kargs.pipeline == 'legacy':
                rpkg, imgOutput = self.render(id, kargs, device=tdev, relighting=False)
                tempImg, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
                loss = self.trainLossLegacy(imgOutput, groundTruth)
            elif kargs.geoloss and iter <= kargs.geolossUntil:
                rpkg, imgOutput, geoOutput = self.renderWithExplicitGeo(
                    id, kargs, device=tdev
                )
                tempImg, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
                if kargs.dsType == 'env':
                    colors = groundTruth[0:3]
                    pos = groundTruth[3:6]
                    bg = groundTruth[6:9]
                    imgOutput = self.mixEnvBG(imgOutput, bg)
                    gtc = t.cat((colors, pos))
                else:
                    gt = groundTruth[0:4]
                    pos = self.datasets[id][1]
                    gtc = torch.cat((gt, pos))
                loss, comp = self.trainLossRelightingGeo(imgOutput, geoOutput, gtc)
            else:
                rpkg, imgOutput = self.render(id, kargs, device=tdev)
                tempImg, vpt, vf, radii = rpkg['render'], rpkg['viewspace_points'], rpkg['visibility_filter'], rpkg['radii']
                if kargs.dsType == 'env':
                    colors = groundTruth[0:3]
                    pos = groundTruth[3:6]
                    bg = groundTruth[6:9]
                    imgOutput = self.mixEnvBG(imgOutput, bg)
                    gt = colors
                else:
                    colorChannelCnt = imgOutput.shape[2]
                    gt = groundTruth[0:colorChannelCnt]
                loss, comp = self.trainLossRelighting(imgOutput, gt)
                
            loss.backward()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iter % 10 == 0:
                    progress_bar.set_postfix(
                        {'Cnt: {} '.format(self.gaussian.xyz.shape[0]) +
                            "Loss": f"{ema_loss_for_log:.{7}f}"
                        })
                    progress_bar.update(10)
                if iter == maxIter:
                    progress_bar.close()

                training_report(
                    tb_writer,
                    iter,
                    loss, comp, None,
                    self, None, None)

                # Log and save
                # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if iter % savePer == 0:
                    print("\n[ITER {}] Saving Gaussians".format(iter))
                    self.save('./furball/pc/{}_pc_{}.ply'.format(trainVersion, iter))

                # Densification
                if iter < kargs.densify_until_iter:
                    # Keep track of max radii in image-space for pruning

                    self.gaussian.max_radii2D[vf] = torch.max(self.gaussian.max_radii2D[vf], radii[vf])
                    self.gaussian.add_densification_stats(vpt, vf)

                    if iter > kargs.densify_from_iter and iter % kargs.densification_interval == 0:
                        # size_threshold = 20 if iter > kargs.opacity_reset_interval else None
                        size_threshold = None
                        self.gaussian.densify_and_prune(kargs.densify_grad_threshold, kargs.opacity_thres, self.radius, size_threshold)
                    
                    if iter % kargs.opacity_reset_interval == 0: # or (dataset.white_background and iter == kargs.densify_from_iter):
                        self.gaussian.reset_opacity()

                # Optimizer step
                if iter < maxIter:
                    self.gaussian.optimizer.step()
                    self.gaussian.optimizer.zero_grad(set_to_none = True)

                if iter % savePer == 0:
                    print("\n[ITER {}] Saving Checkpoint".format(iter))
                    if kargs.geoloss:
                        torch.save(
                            (self.gaussian.capture(), self.aeModel.state_dict(), self.geoModel.state_dict(), iter),
                            "./furball/pc/{}_checkpoint_{}.pth".format(trainVersion, iter))
                    else:
                        torch.save(
                            (self.gaussian.capture(), self.aeModel.state_dict(), iter),
                            "./furball/pc/{}_checkpoint_{}.pth".format(trainVersion, iter))
                 
                    # draw temp picture
                if iter % kargs.trackPer == 0:
                    trackId = kargs.trackId
                    if trackId is not None:
                        trackName1 = './furball/render/{}-{}-loss-{:.3}-track-{}{}.exr'.format(
                            trainVersion, iter, ema_loss_for_log, trackId, ''
                        )
                        trackName2 = './furball/render/{}-{}-loss-{:.3}-track-{}{}.exr'.format(
                            trainVersion, iter, ema_loss_for_log, trackId, '-nnpos'
                        )
                        
                        if kargs.geoloss:
                            _, ans, geo = self.renderWithExplicitGeo(trackId, kargs, device=tdev)
                            if kargs.dsType == 'env':
                                bgTemp = self.datasets[trackId][1][6:9]
                                ans = self.mixEnvBG(ans, bgTemp)
                            imglen = ans.shape[2]
                            geolen = geo.shape[2]
                            saveExr(ans.cpu().numpy(), trackName1, channels=datasetChannels[0:imglen])
                            saveExr(geo.cpu().numpy(), trackName2, channels=datasetChannels[10:10+geolen])
                        elif kargs.pipeline != 'legacy':
                            _, ans = self.render(trackId, kargs, device=tdev)
                            if kargs.dsType == 'env':
                                bgTemp = self.datasets[trackId][1][6:9]
                                ans = self.mixEnvBG(ans, bgTemp)
                            colorCnt = ans.shape[2]
                            if colorCnt > 4:
                                colorCnt = 4
                            saveExr(ans.cpu().numpy()[:,:,0:colorCnt], trackName1, channels=datasetChannels[0:colorCnt])
                        else:
                            pass
                torch.cuda.empty_cache()




def renderSet(scene, kargs : Param, renderList = None):
    with torch.no_grad():
        if scene is None:
            dsList = None
            if kargs.dsType == 'env':
                dsList = readDataset(kargs) # environment force read dataset
            cmrList = readAllConfig(kargs)

            if kargs['checkpoint'] is None:
                kargs['checkpoint'] = './furball/pc/{}_checkpoint_60000.pth'.format(trainVersion)
                kargs['initFile']   = './furball/pc/{}_pc_60000.ply'.format(trainVersion)
            print(tdev)
            gs = GaussianModel(tdev)
            scene = Scene(
                gs,
                kargs,
                cmrList,
                dsList
            )
            scene.setup(kargs)

        gs = scene.gaussian

        if renderList is None:
            print('Train Set Now')
            pb = tqdm(range(1, 2000 + 1), desc='Progress', dynamic_ncols=True)
            for iter in range(0, 2000):
                rpkg, img, geo = scene.renderWithExplicitGeo(iter, kargs, device=tdev)
                if kargs.dsType == 'env':
                    #saveExr(img.cpu().numpy(), './furball/render/{}-train-{}-nobg.exr'.format(trainVersion, iter), datasetChannels[0:4])
                    bgTemp = scene.datasets[iter][1][6:9]
                    img = scene.mixEnvBG(img, bgTemp)
                imgLen = img.shape[2]
                saveExr(img.cpu().numpy(), './furball/render/{}-train-{}.exr'.format(trainVersion, iter), datasetChannels[0:imgLen])
                pb.update(1)
            pb.close()

            print('Test Set Now')
            pb = tqdm(range(1, 100 + 1), desc='Progress', dynamic_ncols=True)
            scene.aeModel.eval()
            for iter in range(2000, 2100):
                rpkg, img, geo = scene.renderWithExplicitGeo(iter, kargs, device=tdev)
                if kargs.dsType == 'env':
                    bgTemp = scene.datasets[iter][1][6:9]
                    img = scene.mixEnvBG(img, bgTemp)
                imgLen = img.shape[2]
                saveExr(img.cpu().numpy()[:,:,:], './furball/render/{}-test-{}.exr'.format(trainVersion, iter), datasetChannels[0:imgLen])
                pb.update(1)
            pb.close()
        else:
            print('Custom Render List')
            pb = tqdm(range(1, len(renderList) + 1), desc='Progress', dynamic_ncols=True)
            for i in renderList:
                if kargs.geoloss:
                    rpkg, img, geo = scene.renderWithExplicitGeo(i, kargs, device=tdev)
                    imgLen = img.shape[2]
                    saveExr(img.cpu().numpy()[:,:,:], './furball/render/{}-RDL-{}.exr'.format(trainVersion, i), datasetChannels[0:imgLen])
                    pb.update(1)
                else:
                    rpkg, img = scene.render(i, kargs, device=tdev)
                    imgLen = img.shape[2]
                    saveExr(img.cpu().numpy()[:,:,:], './furball/render/{}-RDL-{}.exr'.format(trainVersion, i), datasetChannels[0:imgLen])
                    pb.update(1)
            pb.close()

        



def train(kargs : Param):
    # dataset
    renderResult = not kargs['skipResult']
    skipTrain = kargs['skipTrain']
    scene = None

    if not skipTrain:
        cmrList = readAllConfig(kargs)
        dsList = readDataset(kargs)
        gs = GaussianModel(tdev)
        scene = Scene(
            gs,
            kargs,
            cmrList,
            dsList
        )
        scene.setup(kargs)
        scene.startTraining(kargs)

    if renderResult:
        renderSet(scene, kargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Gaussian-AE trainer')
    parser.add_argument('--tag', type=str)
    parser.add_argument('--device', type=int)
    parser.add_argument('--trainSize', type=int)
    parser.add_argument('--testSize', type=int)
    parser.add_argument('--cmrPath', type=str)
    parser.add_argument('--skipTrain', action='store_true')
    parser.add_argument('--skipResult', action='store_true')
    parser.add_argument('--skipVideo', action='store_true')
    parser.add_argument('--skipEval', action='store_true')
    parser.add_argument('--initFile', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--dsName', type=str)
    parser.add_argument('--dsType', type=str)

    return parser.parse_args()


def initEnv(kargs : Param):
    global gpuid
    gpuid = kargs.device
    if gpuid is None:
        raise 'GPUID is None!'
    tdevFormat = 'cuda:{}'

    manual_seed = kargs.seed
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    t.manual_seed(manual_seed)
    assert t.cuda.is_available()
    t.cuda.manual_seed_all(manual_seed)
    t.cuda.set_device(gpuid)

    global tdev
    tdev = tdevFormat.format(gpuid)

    global trainVersion
    if kargs.tag is not None:
        trainVersion = kargs.tag

if __name__ == '__main__':
    args_dict = vars(parse_args())
    args_dict_valid = { key : value for key, value in args_dict.items() if value is not None }
    totalParam.update(args_dict_valid)
    initEnv(totalParam)

    print('TrainVersion: {}'.format(trainVersion))
    print('Device Id: {}'.format(gpuid))
    train(totalParam)
    
    if not totalParam.skipVideo:
        print('\nRendering Video')
        if totalParam.dsType == 'env' or totalParam.dsType == 'enva':
            totalParam['cmrPath'] = './furball/cfgs-envvideo.json'
        else:
            totalParam['cmrPath'] = './furball/cfgs-video.json'
        renderSet(None, totalParam, list(range(0, 360)))
        seq2video = 'ffmpeg -y -gamma 2.2 -r 24 -i ./furball/render/{}-RDL-%1d.exr -vcodec libx265 -framerate 24 -x265-params lossless=1 {}-video.mp4'
        print(os.system(seq2video.format(trainVersion, trainVersion)))

    if not totalParam.skipEval:
        rdl = ['./furball/render/{}-train-{}.exr'.format(trainVersion, i) for i in range(0, 2000)] + \
                ['./furball/render/{}-test-{}.exr'.format(trainVersion, i) for i in range(2000, 2100)]
        gtl = ['./furball/{}/furball-{}-denoised.exr'.format(totalParam.tryGet('dsName', 'gt'), i) for i in range(0, 2100)]
        evaluate(rdl, gtl, outPath=trainVersion+'.txt',dev=tdev)