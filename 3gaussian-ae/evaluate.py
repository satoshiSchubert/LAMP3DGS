import torch
t=torch
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.loss_utils import ssim
from utils.image_utils import psnr
import time

from utils.preprocess import readExr

# To process perf strings
def drawStatisticsFromTXT(path):
    with open(path, 'r') as f:
        strs = f.readlines()
    while strs[-1] == '\n':
        strs.pop()
    iter = 0
    while 'Id' not in strs[iter]:
        iter += 1
    print('Find Statistic Header At Line{}\n'.format(iter))

    ssims = []
    psnrs = []
    lpipss = []
    while iter < len(strs):
        s = [strs[iter+1], strs[iter+2], strs[iter+3]]
        for i in range(3):
            s[i] = float(s[i].split()[1])
        ssims.append(s[0])
        psnrs.append(s[1])
        lpipss.append(s[2])
        iter += 4
    ssims = t.FloatTensor(ssims)
    psnrs = t.FloatTensor(psnrs)
    lpipss = t.FloatTensor(lpipss)
    return ssims, psnrs, lpipss

def evaluate(renderList, gtList, trainCount = 409, outPath = 'log-{}.txt'.format(int(time.time())), dev='cuda:0'):
    cnt = len(renderList)

    print('Running On {}'.format(dev))
    t.cuda.set_device(dev)
    bar = tqdm(range(1, cnt + 1), desc='Progress', dynamic_ncols=True)

    ssims = []
    psnrs = []
    lpipss = []

    for rp, gp in zip(renderList, gtList):
        render = readExr(rp, ['R','G','B'], device=dev).unsqueeze(0)
        ground = readExr(gp, ['R','G','B'], device=dev).unsqueeze(0)
        ssims.append( ssim(render, ground) )
        psnrs.append( psnr(render, ground) )
        lpipss.append( lpips(render, ground, net_type='vgg') )
        bar.update(1)
    bar.close()

    ssims = t.stack(ssims)
    psnrs = t.stack(psnrs)
    lpipss = t.stack(lpipss)

    print('Writing...')
    with open(outPath, 'w') as f:
        f.write('Total:\nSSIM: {}\nPSNR: {}\nLPIPS: {}\n\n\n'.format(
            ssims.mean(),
            psnrs.mean(),
            lpipss.mean()
        ))
        f.write('Train:\nSSIM: {}\nPSNR: {}\nLPIPS: {}\n\n\n'.format(
            ssims[0:trainCount].mean(),
            psnrs[0:trainCount].mean(),
            lpipss[0:trainCount].mean()
        ))
        f.write('Test:\nSSIM: {}\nPSNR: {}\nLPIPS: {}\n\n\n'.format(
            ssims[trainCount:].mean(),
            psnrs[trainCount:].mean(),
            lpipss[trainCount:].mean()
        ))
        for i in range(len(renderList)):
            f.write('Id: {}\nSSIM: {}\nPSNR: {}\nLPIPS: {}\n'.format(
                i,
                ssims[i].item(),
                psnrs[i].item(),
                lpipss[i].item()
            ))
    return ssims, psnrs, lpipss

def generateComparePath(tag, ranges, isRenderList=True):
    renderList = []
    renderBase = './furball/render/{}-{}-{}.exr'
    gtBase = './furball/gt/furball-{}-denoised.exr'
    if isRenderList:
        cnt = len(ranges)
        for i in ranges:
            renderList.append( renderBase.format(tag, 'RDL', i) )
    else:
        trainRange, testRange = ranges
        cnt = len(trainRange) + len(testRange)
        for i in trainRange:
            renderList.append( renderBase.format(tag, 'train', i) )
        for i in testRange:
            renderList.append( renderBase.format(tag, 'test', i) )
    gtList = [gtBase.format(i) for i in range(cnt)]
    return renderList, gtList

if __name__ == '__main__':
    gpuid = 0
    tdev = 'cuda:{}'.format(gpuid)
    torch.cuda.set_device(tdev)
    print('Using Device {}'.format(gpuid))

    trainVersion = '241101_2k_geocheck1'
    renderBase = './furball/render/{}-{}-{}.exr'
    gtBase = './furball/gt/furball-{}-denoised.exr'
    evalRenderList = False

    print('Log File: {}\n'.format(trainVersion+'.txt'))
    # pathList
    if evalRenderList:
        renderList, gtList = generateComparePath(trainVersion, range(0, 2100), True)
    else:
        renderList, gtList = generateComparePath(trainVersion, (range(0, 2000), range(2000, 2100)), False)

    ssims, psnrs, lpips = evaluate(renderList, gtList, outPath=trainVersion+'.txt', dev=tdev)
