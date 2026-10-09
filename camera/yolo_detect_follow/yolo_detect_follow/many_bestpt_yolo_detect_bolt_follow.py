from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO

# ------------------- 1. 加载三个模型 -------------------
MODEL_GRASS = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/bolt.pt"    # bolt
MODEL_TILE  = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/gauge.pt"  # gauge
MODEL_WALL  = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/laying.pt"# laying

CONF_THRESHOLD = 0.1  # 置信度阈值

# -------- 跟随控制参数（bolt跟随）--------
IMG_WIDTH = 1920
IMG_CENTER_X = IMG_WIDTH // 2
AREA_TARGET = 800000
AREA_TOLERANCE = 200000
OFFSET_TOLERANCE = 30
FORWARD_SPEED = 0.3
BACKWARD_SPEED = -0.3
LEFT_SPEED = 0.2
RIGHT_SPEED = -0.2

print("正在加载三个模型...")
model_grass = YOLO(MODEL_GRASS)   # bolt模型，跟随目标
model_tile = YOLO(MODEL_TILE)     # gauge，仅可视化
model_wall = YOLO(MODEL_WALL)     # laying，仅可视化
models = [model_grass, model_tile, model_wall]

# 修复YOLO model.names类型，防止绘图报错
for m in models:
    if isinstance(m.model.names, set):
        m.model.names = {0: list(m.model.names)[0]}
    elif isinstance(m.model.names, list):
        m.model.names = {i: name for i, name in enumerate(m.model.names)}

if __name__ == "__main__":
    # ------------------- 2. 通信、相机、运动客户端初始化 -------------------
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    # 运动控制客户端
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    sport_client.StandUp()
    time.sleep(2)

    # ------------------- 3. 先获取一帧，初始化 code -------------------
    code, data = client.GetImageSample()
    annotated_image = None

    # ------------------- 4. 主循环 -------------------
    while code == 0:
        code, data = client.GetImageSample()
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            continue

        all_detections = []
        bolt_boxes = []  # 专门存放bolt模型输出的框，用于跟随
        for model in models:
            results = model(image, conf=CONF_THRESHOLD, verbose=False)
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                all_detections.append((model, res))
                # 如果是bolt模型，收集它所有检测框
                if model == model_grass:
                    for box in res.boxes:
                        bolt_boxes.append(box)

        annotated_image = image.copy()

        # ========== 运动控制逻辑：只处理bolt ==========
        vx, vy, yaw = 0, 0, 0
        bolt_detected = False
        best_bolt_box = None
        max_area = 0

        if len(bolt_boxes) > 0:
            # 选面积最大bolt框作为跟踪目标
            for box in bolt_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_bolt_box = (x1, y1, x2, y2)

            if best_bolt_box is not None:
                bolt_detected = True
                x1, y1, x2, y2 = best_bolt_box
                box_center_x = (x1 + x2) // 2
                box_area = (x2 - x1) * (y2 - y1)
                offset_x = box_center_x - IMG_CENTER_X

                # 前后控制（像素面积）
                if box_area < (AREA_TARGET - AREA_TOLERANCE):
                    vx = FORWARD_SPEED
                elif box_area > (AREA_TARGET + AREA_TOLERANCE):
                    vx = BACKWARD_SPEED
                # 左右平移控制
                if offset_x > OFFSET_TOLERANCE:
                    vy = RIGHT_SPEED
                elif offset_x < -OFFSET_TOLERANCE:
                    vy = LEFT_SPEED

        # 执行运动：没有bolt就停止
        if not bolt_detected:
            sport_client.StopMove()
        else:
            sport_client.Move(vx, vy, yaw)

        # ========== 绘制全部模型检测框（三类目标全部可视化） ==========
        for model, result in all_detections:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{model.model.names[cls_id]} {conf:.2f}"

                color = (0, 255, 0)
                if model == model_grass:
                    color = (0, 255, 0)   # bolt绿色
                elif model == model_tile:
                    color = (255, 0, 0)   # gauge蓝色
                elif model == model_wall:
                    color = (0, 0, 255)   # laying红色

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("front_camera", annotated_image)
        if cv2.waitKey(20) == 27:  # ESC 退出
            break

    # ------------------- 5. 清理资源 -------------------
    sport_client.StopMove()
    if code != 0:
        print("Get image sample error. code:", code)
    else:
        try:
            cv2.imwrite("front_image.jpg", annotated_image)
        except NameError:
            print("没有保存任何图像，因为从未成功获取到有效帧。")
    cv2.destroyAllWindows()
