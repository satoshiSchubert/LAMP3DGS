Lamp3DGS - 3D Gaussian Sphere
---

# 1. 相机视角生成

参考文件：*1. camera generation*内的cameraGenMy.py，核心是读取pickle文件中的position信息，根据格式要求生成对应的json文件。

生成的json文件路径为:

./3gaussian-ae/furball/cfgs-case.json

# 2. Ground Truth数据集和初始点云生成

参考文件夹：*2. dataset generation*内的initPointMy.py

initPointMy中基于DFS算法根据pickle文件中的world pos buffer生成初始点云point_cloud.ply，并解析pickle得到exr数据集并存放至./3gaussian-ae/furball/中。

可以调节的参数：

```python
l = 2 # 缩放到采样分辨率后的采样距离，表示采样点到给定点的像素距离，后续将连接这两个点作为椭球体的长边

n = 2 # 每一frame采样的高斯球的个数, 所以获得的总高斯球个数为 n*frame_num
```

# 3. 3D GS训练

参考文件夹：*3. gaussian-ae*

主要的库参考environment.yml。如果要从图片生成视频，代码是调ffmpeg，需要额外安装。

所有数据集生成完后放在furball文件夹下。建议ground truth数据集放在./furball/gt-2100/内(furball-[id]-(denoised)?.exr)，相机参数(cfgs-*.json)和初始点云(furball.ply)放在./furball/下即可。

运行之前需要安装submodules/diff-gaussian-rasterization_extend内的库。

```bash
cd ./submodules/diff-gaussian-rasterization_extend
# 过高版本(>=13)的gcc会和cuda冲突，无法编译
export CC=/usr/bin/gcc-12
python setup.py install

# 推荐在visual studio 的prompt环境中激活conda并install
```

命令行参数参考train.py的parse_args，不过通常改全局变量runtimeParam会更快一点。

* cmrPath指定相机参数json。json文件没带分辨率和fov，这部分写在程序内。
* dsName指定数据集文件夹
* dsType指定数据类型(directional | env | enva)，目前有平行光和环境光（分是否带alpha，不带alpha额外需要不带物体的背景图）。不带alpha的版本，3dgs依旧会输出rgba，但是相比其他版本最终会**额外调用Scene::mixEnv函数**与背景图混合，计算rgb\*alpha+bgcolor\*(1-alpha)，得到三通道rgb后做loss。其他版本为四通道rgba的loss。
* tag区分每一次训练和绘制的名字
* device使用的gpuid

train.py程序会：

1. 通过pyexr读数据集为numpy，并pickle dump出去，以后不需要每次都解析exr
2. 读ply并初始化train.py内的Scene（存着大多数数据）
3. 训练完后会生成存着网络的pth和存着高斯数据的ply，默认存在./furball/pc/内
4. 训练完后默认以最新的结果网络绘制训练集和测试集，默认存在./furball/render/内
5. 默认会以cfgs-video.json相机视角生成图片序列，并用ffmpeg转为video
6. 最后会以ssim, psnr, lpips三个metrics评估所有训练测试集，参考evaluate.py内的evaluate函数（其他函数只是辅助函数）。第一次运行lpips相关内容，会自动从github上下载所需模型文件。evaluate函数接受四个参数，分别为：
    * 预测图片的路径的列表(list of string)
    * ground truth的路径(list of string)
    * 打印结果的文本文档路径(string)
    * 使用的device id(string)

e.g.:

```bash
python train.py --device 0 --cmrPath ./furball/cfgs-2100.json --dsName gt-2100 --dsType directional --tag 20241127_test1
```

训练代码主要为train(kargs)，并在其中初始化Scene的对象，调用Scene::startTraining。预测/绘制代码主要为rensetSet(scene, kargs, renderList)。
