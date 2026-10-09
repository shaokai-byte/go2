from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import cv2
import numpy as np
import sys
import time

# ==================== 参数配置 ====================
# 图像参数（Go2 前摄像头 1920x1080）
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
IMG_CENTER_X = IMG_WIDTH // 2

# ROI：只关心画面下半部分（地面），避免远处干扰
ROI_TOP = IMG_HEIGHT // 2
ROI_BOTTOM = IMG_HEIGHT

# 白色阈值（HSV）—— 白线特征：低饱和度、高亮度
WHITE_LOW = np.array([0,   0, 180])
WHITE_HIGH = np.array([180, 60, 255])

# 形态学核大小（去噪）
MORPH_KERNEL = np.ones((7, 7), np.uint8)

# 线检测参数
MIN_LINE_AREA = 3000        # 判定为线的最小像素面积

# 控制参数
FORWARD_SPEED = 0.3         # 前进速度 m/s
YAW_GAIN      = 0.0025      # 偏移量 → 偏航角速度 增益
MAX_YAW       = 0.8         # 偏航角速度上限 rad/s
OFFSET_TOLERANCE = 40       # 中心死区（像素）

# 丢线时的处理
SEARCH_YAW = 0.3            # 丢线时原地慢速旋转寻找
LOST_FRAMES_TO_STOP = 5     # 连续丢线多少帧后停止/进入搜索


# ==================== 白线检测函数 ====================
def detect_white_line(image):
    """
    返回 (cx, cy, area, contour) 或 None
    cx / cy 为整幅图像坐标系下的白线质心
    """
    # 裁剪 ROI
    roi = image[ROI_TOP:ROI_BOTTOM, :]

    # 转 HSV，提取白色
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LOW, WHITE_HIGH)

    # 形态学去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # 找最大白线轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_LINE_AREA:
        return None, mask

    # 计算质心
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"]) + ROI_TOP   # 加回 ROI 偏移

    return (cx, cy, area, largest), mask


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

    # 站立
    sport_client.StandUp()
    time.sleep(2)

    lost_frames = 0
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

        # 白线检测
        result, mask = detect_white_line(image)

        # 画出 ROI 与中心线，方便调试
        cv2.rectangle(image, (0, ROI_TOP), (IMG_WIDTH, ROI_BOTTOM), (255, 128, 0), 2)
        cv2.line(image, (IMG_CENTER_X, ROI_TOP), (IMG_CENTER_X, ROI_BOTTOM), (0, 0, 255), 2)

        if result is not None:
            lost_frames = 0
            cx, cy, area, contour = result

            # 计算水平偏移
            offset_x = cx - IMG_CENTER_X

            # 用偏移量 → 偏航角速度（线在右 → 右转 = 负 yaw）
            if abs(offset_x) > OFFSET_TOLERANCE:
                yaw = -YAW_GAIN * offset_x
                yaw = float(np.clip(yaw, -MAX_YAW, MAX_YAW))
            else:
                yaw = 0.0

            # 前进 + 转向
            sport_client.Move(FORWARD_SPEED, 0.0, yaw)

            # ---- 可视化 ----
            # 把轮廓画到原图（需要加上 ROI_TOP）
            shifted = contour.copy()
            shifted[:, :, 1] += ROI_TOP
            cv2.drawContours(image, [shifted], -1, (0, 255, 0), 3)
            cv2.circle(image, (cx, cy), 10, (0, 255, 255), -1)
            cv2.putText(image,
                        f"offset={offset_x}  yaw={yaw:.2f}  area={int(area)}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        else:
            lost_frames += 1
            if lost_frames <= LOST_FRAMES_TO_STOP:
                # 短暂丢线：继续按上次趋势慢速前进
                sport_client.Move(FORWARD_SPEED * 0.5, 0.0, 0.0)
            else:
                # 长时间丢线：原地慢速旋转搜索
                sport_client.Move(0.0, 0.0, SEARCH_YAW)

            cv2.putText(image, "LINE LOST", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 显示
        cv2.imshow("White Line Follow", image)
        # 可选：显示白色掩膜，调试阈值时打开
        # cv2.imshow("Mask", mask)

        if cv2.waitKey(20) == 27:  # ESC 退出
            break

    # 结束
    sport_client.StopMove()
    cv2.destroyAllWindows()