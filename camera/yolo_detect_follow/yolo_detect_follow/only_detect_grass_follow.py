from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO

# ---------- 1. 模型路径与跟随参数（参考people_follow配置） ----------
MODEL_PATH = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/grass.pt"
CONF_THRESHOLD = 0.1

# 画面参数 Go2前摄像头1920*1080
IMG_WIDTH = 1920
IMG_CENTER_X = IMG_WIDTH // 2

# 距离控制：草地框像素面积，需要根据实际现场调试修改
AREA_TARGET = 800000
AREA_TOLERANCE = 200000

# 水平偏移死区(像素)
OFFSET_TOLERANCE = 30

# 运动速度 m/s
FORWARD_SPEED = 0.3
BACKWARD_SPEED = -0.3
LEFT_SPEED = 0.2
RIGHT_SPEED = -0.2

# ---------- 2. 加载并修复YOLO模型 ----------
model = YOLO(MODEL_PATH)
# 修复model.names类型bug
if isinstance(model.model.names, set):
    model.model.names = {0: 'grass'}
elif isinstance(model.model.names, list):
    model.model.names = {i: name for i, name in enumerate(model.model.names)}

# ---------- 3. 主程序 ----------
if __name__ == "__main__":
    # 初始化通信通道，支持传入机器狗IP
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 视频客户端
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    # 运动控制客户端【新增，来自people_follow】
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    sport_client.StandUp()
    time.sleep(2)  # 等待站立完成

    code, data = client.GetImageSample()
    annotated_image = None

    while code == 0:
        code, data = client.GetImageSample()
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            continue

        results = model(image, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        annotated_image = image.copy()

        # 运动变量初始化
        vx, vy, yaw = 0, 0, 0
        grass_detected = False
        best_box = None
        max_area = 0

        if result.boxes is not None and len(result.boxes) > 0:
            # 遍历所有草地检测框，选取面积最大的框作为跟踪目标
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

            if best_box is not None:
                grass_detected = True
                x1, y1, x2, y2 = best_box
                conf = float(result.boxes[0].conf[0])
                cls_id = int(result.boxes[0].cls[0])
                label = model.model.names.get(cls_id, f"class_{cls_id}")
                text = f"{label} {conf:.2f}"

                # 绘制框与文字
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 计算草地框中心、偏移
                box_center_x = (x1 + x2) // 2
                box_area = (x2 - x1) * (y2 - y1)
                offset_x = box_center_x - IMG_CENTER_X

                # 前后速度控制（面积）
                if box_area < (AREA_TARGET - AREA_TOLERANCE):
                    vx = FORWARD_SPEED
                elif box_area > (AREA_TARGET + AREA_TOLERANCE):
                    vx = BACKWARD_SPEED

                # 左右平移控制（水平偏移）
                if offset_x > OFFSET_TOLERANCE:
                    vy = RIGHT_SPEED
                elif offset_x < -OFFSET_TOLERANCE:
                    vy = LEFT_SPEED

        # 机器人运动决策
        if not grass_detected:
            sport_client.StopMove()
        else:
            sport_client.Move(vx, vy, yaw)

        cv2.imshow("front_camera", annotated_image)
        if cv2.waitKey(20) == 27:
            break

    # 退出处理
    sport_client.StopMove()
    if code != 0:
        print("Get image sample error. code:", code)
    else:
        if annotated_image is not None:
            cv2.imwrite("front_image.jpg", annotated_image)
    cv2.destroyAllWindows()
