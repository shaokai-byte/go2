from ultralytics import YOLO
import yaml
import os

# 创建数据集配置文件时使用绝对路径
dataset_config = {
    'path': os.path.abspath('./grass_dataset'), 
    'train': 'train/images',
    'val': 'val/images',
    'nc': 1,
    'names': {'grass'}
}

config_path = './laying_dataset.yaml'
with open(config_path, 'w') as f:
    yaml.dump(dataset_config, f)

# 初始化模型（选择不同规模的模型）
model = YOLO('yolo11n.pt')  # 官方预训练模型
# 可选模型：yolov11n/s/m/l/x.pt （从轻量到大型）

# 训练参数配置
train_params = {
    'data': config_path,
    'epochs': 100, #训练轮次
    'batch': 4,#批次大小
    'imgsz': 320 * 2,  #图像尺寸
    'device': '0',  # GPU设备ID，CPU使用'cpu'
    'workers': 8,
    'optimizer': 'Adam',  # 可选 SGD/Adam/AdamW等
    'project': './runs/train',  # 训练结果保存目录
    'name': 'grass_detection',
    'exist_ok': True,  # 允许覆盖已有训练结果
    'augment': True,  # 启用数据增强
    #增强参数
    'hsv_h': 0.015,#色相增强
    'hsv_s': 0.7,#饱和度增强
    'hsv_v': 0.4,#明度增强
    'degrees': 10.0,#旋转角度
    'flipud': 0.2,#上下翻转概率
    'fliplr': 0.5,#左右翻转概率
    'mosaic': 1.0,#马赛克增强
    'mixup': 0.1,#混合增强
    'dropout': 0.1,#dropout层概率
    #学习率
    'lr0': 0.001,      # 初始学习率  
    'lrf': 0.1,       # 最终学习率 = lr0 * lrf
    'save_period': 5, # 每10个epoch保存一次模型
    'amp': True,       # 启用自动混合精度训练 
    'patience': 0    #早停机制
} 

# 开始训练
results = model.train(**train_params)

# 验证最佳模型
best_model = YOLO(os.path.join(train_params['project'], 
                train_params['name'], 'weights/best.pt'))
metrics = best_model.val(
    data = config_path,
    batch = 32,
    conf = 0.001,  # 验证置信度阈值
    iou = 0.6,      # NMS IoU阈值
    exist_ok = True
)



