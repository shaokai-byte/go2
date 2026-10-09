# coco_ssd_detect
本文件夹主要是通过SSD深度学习目标检测模型和coco数据集库完成目标物体检测和识别人体跟随
已实现目标物体检测和人体跟随，后续可在人体跟随的程序和COCO中已有的训练物体基础上尝试其他物体跟随，也可从0完整完成SSD模型训练（需要自己在SSD开源仓库中自行学习）

## 环境
unitree_sdk2_python基础环境
opencv库安装（大部分自有）

## 数据模型文件
存放ssd模型文件
## 程序复件
备份程序
##coco_ssd_object_detection.py
目标检测程序
##people_detect_follow.py
识别人体跟随程序

##使用方法

终端在该文件夹目录文件下python3运行即可
# python3 coco_ssd_object_detection.py
# python3 people_detect_follow.py

##注意
两个程序必须和模型在同一个文件夹下
