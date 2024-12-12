import json
import math
import numpy as np
import torch
from torch import nn
import Imath
import OpenEXR
import array
from plyfile import PlyData, PlyElement

t = torch

def readExr(path : str, channels = ['R', 'G', 'B'], transposed = True, device = 'cpu') -> t.tensor:
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Assuming the EXR image has RGB channels
    img = np.zeros((len(channels), height, width), dtype=np.float32)

    for i, channel in enumerate(channels):
        img[i, :, :] = np.frombuffer(exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape((height, width))

    if not transposed:
        img = img.transpose(1, 2, 0)
    imgTensor = t.from_numpy(img).to(device)

    return imgTensor

def saveExr(pic, path : str, channels = ['R', 'G', 'B'], transposed = False):
    assert len(pic.shape) == 3 or (len(pic.shape)==4 and pic.shape[0] == 1)
    if len(pic.shape)==4:
        pic = pic.squeeze(0)
    if not transposed:
        if type(pic) == np.array or type(pic) == np.ndarray:
            pic = pic.transpose(2, 0, 1)
        else:
            pic = pic.permute(2, 0, 1)
    assert pic.shape[0] == len(channels)

    height = pic.shape[1]
    width = pic.shape[2]

    hd = OpenEXR.Header(width, height)
    tempChannels = {}
    for i in channels:
        tempChannels[i] = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    hd['channels'] = tempChannels
    file = OpenEXR.OutputFile(path, hd)
    tempPic = pic.reshape(len(channels), -1)

    ansDict = {}
    for i in range(pic.shape[0]):
        ansDict[ channels[i] ] = array.array('f', tempPic[i, :]).tobytes()
        
    file.writePixels(ansDict)


def loadPly(path : str):
    plydata = PlyData.read(path)
    count = plydata.elements[0].count
    props = plydata.elements[0].properties
    propNum = len(props)

    ans = np.zeros((count, propNum), dtype=np.float32)
    for i, prop in enumerate(props):
        ans[:, i] = np.asarray(plydata.elements[0][prop])

    return ans, props

def savePly(data, props, path : str):
    dtype_full = [(attribute, 'f4') for attribute in props]
    data = data.reshape((data.shape[0], -1))
    elements = np.zeros(data.shape[0], dtype=dtype_full)
    for i, prop in enumerate(props):
        elements[prop] = data[:, i]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def getPathList(paths, device : str):
    pathList = None
    id = None
    if callable(paths):
        id, pathList = paths()
    else:
        assert type(paths) == list
        if len(paths) > 0 and type(paths[0]) == tuple and type(paths[0][0]) == int:
            id, pathList = zip(*paths)
        else:
            id = [i for i in range(len(paths))]
            pathList = paths
    return id, pathList



