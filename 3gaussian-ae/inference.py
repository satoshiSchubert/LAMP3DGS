import os.path

from train import *

def render3DGS(scene, kargs : Param, result_folder, renderList = None):
    if not os.path.exists(f"./inference/{result_folder}"):
        os.makedirs(f"./inference/{result_folder}")
    print(" ------ 测试render时间 ------ ")
    with torch.no_grad():
        testsize = 256 # 从cfg中读取：camera的个数
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True)
        if renderList is None:
            print('Start Now')
            pb = tqdm(range(1, testsize + 1), desc='Progress', dynamic_ncols=True)

            # 测试渲染时间
            starter.record()
            for iter in range(0, testsize):
                rpkg, img, geo = scene.renderWithExplicitGeo(iter, kargs, device=tdev)
                imgLen = img.shape[2]
                saveExr(img.cpu().numpy(),
                        f'./inference/{result_folder}/result_{iter}.exr',
                        datasetChannels[0:imgLen])
                pb.update(1)
            ender.record()
            print(f'Time per iter: {starter.elapsed_time(ender)/testsize} ')
            pb.close()

# load 3dgs and render exr according to json file.
if __name__ == '__main__':
    args_dict = vars(parse_args())
    args_dict_valid = {key: value for key, value in args_dict.items() if value is not None}
    totalParam.update(args_dict_valid)
    initEnv(totalParam)

    # 新建文件夹
    if not os.path.exists(f"./inference/"):
        os.makedirs(f"./inference/")

    # 3dgs path:
    gs_path = f"./inference/20250110v3shtcase1_pc_700000.ply"
    # inference camera cfg path:
    cmr_cfg_path = f"./inference/cmrcfgs.json"

    # load 3dgs and cmr dirs, and construct scene object
    cmrList = readAllConfig(totalParam)
    dsList = readDataset(totalParam)

    print(tdev)
    gs = GaussianModel(tdev)
    gs.loadPly(gs_path)
    scene = Scene(
        gs,
        totalParam,
        cmrList,
        dsList
    )
    #scene.setup(totalParam)

    render3DGS(scene, totalParam,"result_sht")






