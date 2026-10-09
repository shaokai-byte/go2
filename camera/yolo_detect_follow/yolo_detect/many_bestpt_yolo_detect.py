from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys
from ultralytics import YOLO

# ------------------- 1. 加载三个模型 -------------------
MODEL_GRASS = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/bolt.pt"
MODEL_TILE  = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/gauge.pt"
MODEL_WALL  = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/laying.pt"

CONF_THRESHOLD = 0.1  # 置信度阈值

print("正在加载三个模型...")
model_grass = YOLO(MODEL_GRASS)
model_tile = YOLO(MODEL_TILE)
model_wall = YOLO(MODEL_WALL)
models = [model_grass, model_tile, model_wall]  # 保留列表以便后续遍历

if __name__ == "__main__":
    # ------------------- 2. 相机初始化 -------------------
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    # ------------------- 3. 先获取一帧，初始化 code -------------------
    code, data = client.GetImageSample()

    # ------------------- 4. 主循环 -------------------
    while code == 0:
        # 获取图像
        code, data = client.GetImageSample()
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # 对每个模型进行推理，并保存结果及其对应的模型对象
        all_detections = []  # 每个元素为 (模型对象, 检测结果)
        for model in models:
            results = model(image, conf=CONF_THRESHOLD)
            # 如果检测到目标，则保存
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                all_detections.append((model, results[0]))

        # 绘制所有检测结果
        annotated_image = image.copy()
        for model, result in all_detections:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                # 从该模型的类别名称中获取标签
                label = f"{model.names[cls_id]} {conf:.2f}"

                # 用不同颜色区分不同模型（可自定义）
                color = (0, 255, 0)  # 默认绿色
                if model == model_grass:
                    color = (0, 255, 0)   # 绿色
                elif model == model_tile:
                    color = (255, 0, 0)   # 蓝色
                elif model == model_wall:
                    color = (0, 0, 255)   # 红色

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("front_camera", annotated_image)

        if cv2.waitKey(20) == 27:  # ESC 退出
            break

    # ------------------- 5. 清理资源 -------------------
    if code != 0:
        print("Get image sample error. code:", code)
    else:
        # 保存最后一张标注图像（如果 annotated_image 已定义）
        try:
            cv2.imwrite("front_image.jpg", annotated_image)
        except NameError:
            print("没有保存任何图像，因为从未成功获取到有效帧。")

    cv2.destroyAllWindows()