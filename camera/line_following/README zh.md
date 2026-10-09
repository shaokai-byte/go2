# line_following
该文件夹主要是5个巡线行走程序
核心是利用opencv自带的颜色库再调用机器狗行走接口程序
已实现巡线行走并调取机器狗动作，后续可继续优化巡线算法，巡走颜色线可自行训练识别效果更好

## 环境
unitree_sdk2_python基础环境
opencv库安装（大部分自有）

## 程序复件
备份程序
## red_line.py
识别红线行走
## red_line_down.py
识别红线完成后坐下再起立
## red_line_sit_keep.py
识别红线完成后保持坐下
## white_line.py
识别白线行走
## white_line_sit_keep.py
识别白线完成后保持坐下

##使用方法
终端在该文件夹目录文件下python3运行即可
# python3 red_line.py
# python3 red_line_down.py
.....

##注意
运行程序后需要按遥控器start键才开始巡线
