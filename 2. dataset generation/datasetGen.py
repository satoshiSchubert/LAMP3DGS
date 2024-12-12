import mitsuba as mi
import numpy as np
import random
import json
from math import *
from tqdm import tqdm
import os
import gc

mi.set_variant("cuda_ad_rgb")



def genDataset(scenePath, cmrCfg, resultPrefix):
    try:
        os.mkdir(resultPrefix)
    except OSError as error:
        pass
    
    scene = mi.load_file(scenePath)
    with open(cmrCfg, 'r') as f:
        ans = json.load(f)

    params = mi.traverse(scene)

    swapMat = mi.Transform4f([ [0, -1, 0, 0], [1, 0, 0, 0], [0,0,1,0],[0,0,0,1] ])

    if 'direction' in ans[0]:
        pb = tqdm(range(1, len(ans)+1), desc='Directional', dynamic_ncols=True)
        for i in ans:
            viewMat = i['viewMat']
            direction = i['direction']
            temp = mi.Transform4f.from_frame( mi.Frame3f(direction) )
            directMat = (swapMat @ temp).inverse()

            params['PerspectiveCamera.to_world'] = mi.Transform4f(viewMat)
            params['DirectionalEmitter.to_world'] = directMat
            params.update()

            to_sensor = scene.sensors()[0].world_transform().inverse()

            _ = mi.render(scene)
            img = scene.sensors()[0].film().bitmap()

            denoiser = mi.OptixDenoiser(input_size=img.size(), albedo=True, normals=True, temporal=False)
            denoised = denoiser(img, albedo_ch='alb', normals_ch='nn', to_sensor=to_sensor)
            
            img.write("./{}/furball-{}.exr".format(resultPrefix, i['id']))
            denoised.write("./{}/furball-{}-denoised.exr".format(resultPrefix, i['id']))

            pb.update(1)
        pb.close()
    else:
        assert 'envRot' in ans[0]
        
        pb = tqdm(range(1, len(ans)+1), desc='Env', dynamic_ncols=True)

        for i in ans:
            viewMat = i['viewMat']

            envRot = i['envRot']

            params['PerspectiveCamera.to_world'] = mi.Transform4f(viewMat)

            theta = envRot / 180 * 3.14159265359
            rot = mi.Transform4f([ [cos(theta), 0, -sin(theta), 0], [0,1,0,0], [sin(theta), 0, cos(theta), 0], [0,0,0,1] ])

            params['EnvironmentMapEmitter.to_world'] = rot
            params.update()

            to_sensor = scene.sensors()[0].world_transform().inverse()

            _ = mi.render(scene)
            img = scene.sensors()[0].film().bitmap()
            img.write("./{}/furball-{}.exr".format(resultPrefix, i['id']))
            
            if scenePath != './scene_bg.xml':
                # rendering only envmap does not have albedo
                denoiser = mi.OptixDenoiser(input_size=img.size(), albedo=True, normals=True, temporal=False)
                denoised = denoiser(img, albedo_ch='alb', normals_ch='nn', to_sensor=to_sensor)
                denoised.write("./{}/furball-{}-denoised.exr".format(resultPrefix, i['id']))

            pb.update(1)
        pb.close()

if __name__ == '__main__':
    genDataset('./scene.xml', './cfgs-2100.json', 'gt')
    genDataset('./scene.xml', './cfgs-video.json', 'gt-video')

    genDataset('./scene_enva.xml', './cfgs-env.json', 'gt-enva')
    genDataset('./scene_enva.xml', './cfgs-envvideo.json', 'gt-envavideo')

    genDataset('./scene_env.xml', './cfgs-env.json', 'gt-env')
    genDataset('./scene_bg.xml', './cfgs-env.json', 'gt-envbg')

    genDataset('./scene_env.xml', './cfgs-envvideo.json', 'gt-envvideo')
    genDataset('./scene_bg.xml', './cfgs-envvideo.json', 'gt-envvideobg')