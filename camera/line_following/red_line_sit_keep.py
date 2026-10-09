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

# 红色阈值（HSV）—— 红色跨 0°/180°，需要两段
RED_LOW1  = np.array([0,   80,  60])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 80,  60])
RED_HIGH2 = np.array([180, 255, 255])

# 形态学核大小（去噪）
MORPH_KERNEL = np.ones((7, 7), np.uint8)

# 线检测参数
MIN_LINE_AREA = 3000        # 判定为线的最小像素面积

# 控制参数
FORWARD_SPEED = 0.5         # 前进速度 m/s
YAW_GAIN      = 0.0025      # 偏移量 → 偏航角速度 增益
MAX_YAW       = 0.8         # 偏航角速度上限 rad/s
OFFSET_TOLERANCE = 40       # 中心死区（像素）

# 丢线时的处理
SEARCH_YAW = 0.3            # 丢线时原地慢速旋转寻找
LOST_FRAMES_TO_STOP = 5     # 连续丢线多少帧后进入搜索

# ==================== 结束逻辑参数 ====================
FINAL_FORWARD_DISTANCE = 2.5   # 识别结束后再前进的距离（米）
LINE_LOST_FRAMES_FINAL = 50    # 连续丢线超过此帧数视为红线结束
FINAL_MOVE_PERIOD = 0.1        # 每步下发速度命令的周期（秒）

SIT_HOLD_SECONDS = 3.0         # 坐下后保持坐姿的时间（秒），期间持续下发 Sit
SIT_HOLD_PERIOD  = 0.5         # 保持坐姿时重复下发 Sit 的周期（秒）


# ==================== 红线检测函数 ====================
def detect_red_line(image):
    """返回 (cx, cy, area, contour) 或 None"""
    roi = image[ROI_TOP:ROI_BOTTOM, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_LINE_AREA:
        return None, mask

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"]) + ROI_TOP

    return (cx, cy, area, largest), mask


# ==================== 结束流程：前进 2.5 米 + 坐下并保持 ====================
def finish_sequence(sport_client):
    """
    红线识别结束后：
      1) 再前进 FINAL_FORWARD_DISTANCE 米
      2) 停止移动
      3) 执行 Sit()，并保持坐姿
    """
    # ---- 1) 前进固定距离 ----
    duration = FINAL_FORWARD_DISTANCE / abs(FORWARD_SPEED)
    print(f"[FINISH] 前进 {FINAL_FORWARD_DISTANCE} 米（约 {duration:.2f} 秒）...")

    t0 = time.time()
    while time.time() - t0 < duration:
        sport_client.Move(FORWARD_SPEED, 0.0, 0.0)
        time.sleep(FINAL_MOVE_PERIOD)

    # ---- 2) 停止移动（只发一次，避免和后面的 Sit 冲突）----
    sport_client.StopMove()
    time.sleep(0.5)

    # ---- 3) 坐下并保持坐姿 ----
    print("[FINISH] 执行坐下动作 Sit() ...")
    sport_client.Sit()
    time.sleep(1.5)   # 等待坐下动作完成

    # 保持阶段：周期性地重复下发 Sit()，防止固件超时恢复站立
    print(f"[FINISH] 保持坐姿 {SIT_HOLD_SECONDS} 秒 ...")
    t0 = time.time()
    while time.time() - t0 < SIT_HOLD_SECONDS:
        sport_client.Sit()
        time.sleep(SIT_HOLD_PERIOD)

    print("[FINISH] 完成，机器人保持坐姿。")


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

    # 状态变量
    line_lost_count = 0        # 连续丢线计数
    finish_triggered = False   # 是否进入结束流程

    code, data = video_client.GetImageSample()

    while code == 0 and not finish_triggered:
        code, data = video_client.GetImageSample()
        if code != 0:
            break

        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            continue

        result, mask = detect_red_line(image)

        cv2.rectangle(image, (0, ROI_TOP), (IMG_WIDTH, ROI_BOTTOM), (255, 128, 0), 2)
        cv2.line(image, (IMG_CENTER_X, ROI_TOP), (IMG_CENTER_X, ROI_BOTTOM), (0, 0, 255), 2)

        if result is not None:
            line_lost_count = 0
            cx, cy, area, contour = result

            offset_x = cx - IMG_CENTER_X
            if abs(offset_x) > OFFSET_TOLERANCE:
                yaw = float(np.clip(-YAW_GAIN * offset_x, -MAX_YAW, MAX_YAW))
            else:
                yaw = 0.0

            sport_client.Move(FORWARD_SPEED, 0.0, yaw)

            shifted = contour.copy()
            shifted[:, :, 1] += ROI_TOP
            cv2.drawContours(image, [shifted], -1, (0, 255, 0), 3)
            cv2.circle(image, (cx, cy), 10, (0, 255, 255), -1)
            cv2.putText(image,
                        f"offset={offset_x}  yaw={yaw:.2f}  area={int(area)}",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        else:
            line_lost_count += 1

            if line_lost_count >= LINE_LOST_FRAMES_FINAL:
                print(f"[INFO] 连续丢线 {line_lost_count} 帧，判定红线识别结束。")
                finish_triggered = True
                break

            elif line_lost_count <= LOST_FRAMES_TO_STOP:
                sport_client.Move(FORWARD_SPEED * 0.5, 0.0, 0.0)
            else:
                sport_client.Move(0.0, 0.0, SEARCH_YAW)

            cv2.putText(image, "LINE LOST", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Red Line Follow", image)
        # cv2.imshow("Mask", mask)

        if cv2.waitKey(20) == 27:
            print("[INFO] 用户按 ESC，进入结束流程。")
            finish_triggered = True
            break

    # ==================== 结束流程 ====================
    if finish_triggered:
        try:
            finish_sequence(sport_client)
        except Exception as e:
            print(f"[ERROR] 结束流程异常: {e}")
            # 异常兜底：至少让机器人停下（此时未成功 Sit，才调 StopMove）
            try:
                sport_client.StopMove()
            except Exception:
                pass
    else:
        # 视频流异常退出等非正常情况：停止运动
        try:
            sport_client.StopMove()
        except Exception:
            pass

    # ⚠️ 注意：finish_sequence 成功执行后不再下发任何 Move/StopMove/StandUp
    #         机器人保持坐姿，脚本退出。
    cv2.destroyAllWindows()
    print("[DONE] 脚本退出，机器人保持坐姿。")
