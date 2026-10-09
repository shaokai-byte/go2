from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import cv2
import numpy as np
import sys
import time

# ==================== 参数配置 ====================
TARGET_CLASS = "PERSON"
CONFIDENCE_THRES = 0.6

# 画面参数（Go2 前摄像头 1920x1080）
IMG_WIDTH = 1920
IMG_CENTER_X = IMG_WIDTH // 2

# 距离控制（检测框面积）
AREA_TARGET = 800000   # 目标面积（根据实际调试修改）
AREA_TOLERANCE = 200000  # 面积容差（死区）

# 方向控制（水平偏移像素）
OFFSET_TOLERANCE = 30

# 运动参数（m/s）
FORWARD_SPEED = 0.2
BACKWARD_SPEED = -0.15
LEFT_SPEED = 0.15
RIGHT_SPEED = -0.15

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 初始化通信通道
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 初始化视频客户端
    video_client = VideoClient()
    video_client.SetTimeout(60.0)
    video_client.Init()

    # 初始化运动控制客户端
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # 确保机器人处于站立状态
    sport_client.StandUp()
    time.sleep(2)

    # 加载 COCO 类别名称
    classNames = []
    classFile = "coco.names"
    try:
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
    except FileNotFoundError:
        print(f"错误：未找到 {classFile}")
        sys.exit(1)

    # 加载 SSD 模型
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"
    try:
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
    except Exception as e:
        print(f"模型加载失败：{e}")
        sys.exit(1)

    # 循环检测与跟随
    code, data = video_client.GetImageSample()
    while code == 0:
        code, data = video_client.GetImageSample()
        if code != 0:
            break

        # 解码图像
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            continue

        # 目标检测
        classIds, confidences, bboxes = net.detect(image, confThreshold=CONFIDENCE_THRES)

        # 默认不移动
        vx, vy, yaw = 0, 0, 0
        human_detected = False

        # 查找最匹配的人体
        if len(classIds) > 0:
            max_area = 0
            best_bbox = None
            for classId, conf, box in zip(classIds.flatten(), confidences.flatten(), bboxes):
                if (classId - 1) < len(classNames) and classNames[classId - 1].upper() == TARGET_CLASS:
                    area = box[2] * box[3]
                    if area > max_area:
                        max_area = area
                        best_bbox = box

            if best_bbox is not None:
                human_detected = True
                x_min, y_min, width, height = best_bbox
                center_x = x_min + width // 2
                area = width * height

                # 计算偏移
                offset_x = center_x - IMG_CENTER_X

                # 显示检测框
                cv2.rectangle(image, best_bbox, color=(0, 255, 0), thickness=2)
                cv2.putText(image, "PERSON", (x_min + 10, y_min + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 根据面积判断前后
                if area < (AREA_TARGET - AREA_TOLERANCE):
                    vx = FORWARD_SPEED
                elif area > (AREA_TARGET + AREA_TOLERANCE):
                    vx = BACKWARD_SPEED

                # 根据偏移判断左右
                if offset_x > OFFSET_TOLERANCE:
                    vy = RIGHT_SPEED
                elif offset_x < -OFFSET_TOLERANCE:
                    vy = LEFT_SPEED

        # 如果没有检测到人体则停止
        if not human_detected:
            sport_client.StopMove()
        else:
            sport_client.Move(vx, vy, yaw)

        # 显示画面
        cv2.imshow("Human Follow", image)
        if cv2.waitKey(20) == 27:  # ESC 退出
            break

    # 结束
    sport_client.StopMove()
    cv2.destroyAllWindows()
