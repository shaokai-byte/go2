# yolo_detect_follow
目标物体检测识别后跟随两个程序
已完成模型文件中的模型识别跟随，后续可继续调用识别其他目标检测物体并跟随

## 环境
unitree_sdk2_python基础环境
opencv库安装（大部分自有）

## many_bestpt_yolo_detect_bolt_follow.py
识别bolt物体后跟随
## only_detect_grass_follow.py
识别grass物体后跟随

##使用方法
终端在该文件夹目录文件下python3运行即可
# python3 many_bestpt_yolo_detect_bolt_follow.py
# python3 only_detect_grass_follow.py

##注意
程序中调用模型的路径是绝对路径，运行时要改为自己电脑中模型所在的绝对路径
