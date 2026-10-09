# camera
本文件夹基于unitree官网python服务接口unitree_sdk2_python文件夹中front_camera文件中的程序，在调用机器狗相机的基础上，赋予其相机目标检测，目标识别并在识别的基础上调用行走，动作的功能

## 前提
已完成官网（https://support.unitree.com/home/zh/developer/Python）python服务接口环境部署 ，并安装unitree_sdk2_python
已成功运行unitree_sdk2_python/example/go2/high_level文件夹中的程序
已成功运行unitree_sdk2_python/example/go2/front_camera文件夹中的程序
已成功运行sport文件夹中的程序

## coco_ssd_detect
基于ssd模型和coco数据集的目标检测和物体跟随

## gesture_detect_control
基于Mediapipe库的手势识别和机器狗动作控制

## line_following 
基于opencv库的巡线行走

## yolo_best.pt
训练好的yolo模型文件

## yolo_detect_follow
基于yolo深度学习的目标检测和物体跟随


