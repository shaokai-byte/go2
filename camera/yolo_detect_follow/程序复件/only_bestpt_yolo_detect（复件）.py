from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys
from ultralytics import YOLO

# ---------- 1. 模型路径与参数 ----------
MODEL_PATH = "/home/s/paddle-robotdog-course-full/unitree_sdk2_python/example/go2/front_camera/yolo_best/grass.pt"
CONF_THRESHOLD = 0.1

# ---------- 2. 加载并修复模型 ----------
model = YOLO(MODEL_PATH)

# *** 关键修复：修正模型内部的 names（可写）***
# 检查并转换 names
if isinstance(model.model.names, set):
    # 如果你的模型只有一个类别，直接设为 {'0': 'grass'}
    model.model.names = {0: 'grass'}
    # 如果多个类别，需要根据训练时的顺序，这里假设只有一个类别
elif isinstance(model.model.names, list):
    model.model.names = {i: name for i, name in enumerate(model.model.names)}
# 如果已经是 dict 则无需操作

# （可选）也可以全局禁用详细日志输出
# model.predictor.args.verbose = False   # 但此属性可能不存在，故在调用时加 verbose=False

# ---------- 3. 主程序 ----------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    code, data = client.GetImageSample()
    annotated_image = None

    while code == 0:
        code, data = client.GetImageSample()
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            continue

        # 推理时添加 verbose=False，避免日志生成时再次访问 names
        results = model(image, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        # 手动绘制检测框（避免依赖 plot() 内部的 verbose）
        annotated_image = image.copy()
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                # 使用自定义类别映射（如果希望显示名称）
                label = model.model.names.get(cls_id, f"class_{cls_id}")
                text = f"{label} {conf:.2f}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("front_camera", annotated_image)
        if cv2.waitKey(20) == 27:
            break

    if code != 0:
        print("Get image sample error. code:", code)
    else:
        if annotated_image is not None:
            cv2.imwrite("front_image.jpg", annotated_image)

    cv2.destroyAllWindows()