from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
import cv2
import numpy as np
import sys


if __name__ == "__main__":
    # 初始化机器人通信通道（支持命令行传网络接口，如 python 脚本.py wlan0）
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 创建并初始化视频客户端
    client = VideoClient()
    client.SetTimeout(60.0)
    client.Init()

    # ---------------------- 加载目标检测相关资源 ----------------------
    # 加载 COCO 类别名称
    classNames = []
    classFile = "coco.names"
    try:
        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
    except FileNotFoundError:
        print(f"错误：未找到 {classFile}，请确保文件在代码目录下。")
        sys.exit(1)

    # 加载 SSD 目标检测模型
    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"
    try:
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        # 设置模型输入参数（与训练时一致）
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)  # BGR 转 RGB（模型训练用 RGB）
    except Exception as e:
        print(f"模型加载失败：{e}")
        print("请确保模型权重和配置文件在代码目录下。")
        sys.exit(1)

    # ---------------------- 循环获取图像并检测 ----------------------
    code, data = client.GetImageSample()
    while code == 0:  # code==0 表示图像获取正常
        code, data = client.GetImageSample()
        if code != 0:
            break

        # 图像二进制数据转 OpenCV 格式
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            print("图像解码失败，跳过当前帧。")
            continue

        # 执行目标检测（置信度阈值设为 0.5，可根据需求调整）
        conf_threshold = 0.5
        classIds, confidences, bboxes = net.detect(image, confThreshold=conf_threshold)

        # 绘制检测结果（边界框、类别名、置信度）
        if len(classIds) > 0:
            for classId, conf, box in zip(classIds.flatten(), confidences.flatten(), bboxes):
                # 画边界框
                cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
                # 显示类别名称
                class_name = classNames[classId - 1].upper() if (classId - 1) < len(classNames) else "Unknown"
                cv2.putText(
                    image, class_name,
                    (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                # 显示置信度
                cv2.putText(
                    image, f"{conf * 100:.2f}%",
                    (box[0] + 10, box[1] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        # 显示带检测结果的图像
        cv2.imshow("front_camera (with detection)", image)
        # 按 ESC（键码 27）退出循环
        if cv2.waitKey(20) == 27:
            break

    # ---------------------- 结束处理 ----------------------
    if code != 0:
        print("获取图像失败，错误码：", code)
    else:
        # 保存最后一帧（带检测结果）
        cv2.imwrite("detected_front_image.jpg", image)

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()
