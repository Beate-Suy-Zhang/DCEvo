# DCEvo

Jinyuan Liu, Bowei Zhang, Qingyun Mei, Xingyuan Li, Yang Zou, Zhiying Jiang, Long Ma, Risheng Liu, Xin Fan, **"DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion"**,
IEEE/CVF Conference on Computer Vision and Pattern Recognition **(CVPR)**, 2025.

![Abstract](Figure/first_figure.jpg)


## 数据集准备
我们在 **"DCEvo/datasets"** 中提供了数据集的示例。

我们测试图像融合依据 [这个项目](https://github.com/RollingPlain/IVIF_ZOO/) 的完整数据集。


## 测试图像融合  
我们的模型参数被保存在 **"DCEvo/ckpt"**。 然后，可以通过以下代码测试我们的纯融合方法：
```
python test_Fusion.py
```


## 上色灰度图像
可以通过下方代码上色灰度图片用于任务引导的图像融合训练和测试：
```
python tocolor.py
```


## 融合结果
1. 我们的 DCEvo 和最先进的方法之间在 M3FD、RoadScene、TNO 和 FMB 数据集上红外和可见光图像融合的定量比较。
![Abstract](Figure/Quantitative_Fusion.png)

2. 我们的 DCEvo 和现有图像融合方法的可视化比较。从上到下：TNO 中的低光图像，RoadScene 中的高亮度图像，M3FD 中的低质量图像。
![Abstract](Figure/fusionresult.png)


## 测试任务驱动的图像融合  
测试需要 **"DCEvo/datasets/M3FD/images"** 中生成 **RGB 纯融合** 图像。
通过以下代码，可以测试任务引导的图像融合：
```
python test_task_guided_fusion.py
```


## 任务驱动的下游 IVIF 应用结果
1. 我们的 DCEvo 和现有的图像融合方法在 M3FD 和 FMB 数据集上用于下游检测和分割任务的定量比较。
![Abstract](Figure/Quantitative_Task.png)

2. 在 M3FD 数据集上，对我们的方法和现有的红外和可见光图像融合方法在下游目标检测中的定可视化较任务。我们的融合图像中的物体被完全检测到。
![Abstract](Figure/Detect.png)

3. 我们的 DCEvo 与 FMB 数据集上不同融合方法生成的融合图像的可视化比较。我们的方法在烟雾场景和白天场景中执行最佳分割结果。
![Abstract](Figure/Segment2.png)


## 训练   
训练此过程需要先在 **"DCEvo/datasets/M3FD/images"** 中生成 **RGB 纯融合** 图像。
通过以下代码，可以训练任务引导的图像融合：
```
python DCEvo_train.py
```
