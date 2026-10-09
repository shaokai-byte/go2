import time
import sys
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Value
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
import mediapipe as mp_mediapipe
import signal

# 机器人运动控制类（适配 Go2 EDU 动作接口）
class SportModeTest:
    def __init__(self) -> None:
        try:
            self.client = SportClient()
            self.client.SetTimeout(10.0)
            self.client.Init()
            print("SportClient 初始化成功")
        except AttributeError as e:
            print(f"SportClient 初始化失败: {e}")
        except Exception as e:
            print(f"运动控制客户端异常: {e}")

    def Stand(self):
        try:
            self.client.StandUp()
            print("机器人执行：站立")
        except Exception as e:
            print(f"站立动作执行失败: {e}")

    def Sit(self):
        try:
            self.client.Sit()
            print("机器人执行：坐下")
        except Exception as e:
            print(f"坐下动作执行失败: {e}")

    def Hello(self):
        try:
            self.client.Hello()
            print("机器人执行：打招呼")
        except Exception as e:
            print(f"打招呼动作执行失败: {e}")

    def Heart(self):
        try:
            self.client.Heart()
            print("机器人执行：比爱心")
        except Exception as e:
            print(f"比爱心动作执行失败: {e}")

# 初始化视频客户端（获取机器狗前置摄像头画面）
def initialize_video_client(network_interface):
    try:
        ChannelFactoryInitialize(0, network_interface)
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()
        print("视频客户端初始化成功")
        return client
    except Exception as e:
        print(f"视频客户端初始化失败: {e}")
        return None

# 获取视频帧（解码机器狗摄像头数据）
def get_video_frame(client):
    if client is None:
        return None
    try:
        code, data = client.GetImageSample()
        if code == 0:
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"获取视频帧失败，错误码: {code}")
            return None
    except Exception as e:
        print(f"视频帧解码异常: {e}")
        return None

# 绘制手势检测区域（屏幕中央检测框）
def draw_detection_area(image, radius=200):
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    # 绘制检测区域圆
    cv2.circle(image, center, radius, (0, 255, 0), 3)
    cv2.putText(image, "Gesture Detection Area", (center[0]-150, center[1]-radius-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # 创建掩码（仅检测圆形区域内的手势）
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask

# 手势检测进程（独立线程处理摄像头画面）
def gesture_detection_process(pipe, network_interface, last_action_time_shared):
    # 初始化视频客户端
    client = initialize_video_client(network_interface)
    if client is None:
        print("视频客户端初始化失败，退出手势检测进程")
        return

    # 初始化 MediaPipe 手势检测
    mp_hands = mp_mediapipe.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 仅检测一只手（简化控制逻辑）
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    drawing_utils = mp_mediapipe.solutions.drawing_utils

    # 手势状态变量
    gesture_last_sent = None
    gesture_labels = ["无", "1指", "2指", "3指", "4指", "5指", "6指", "7指", "8指", "9指", "10指"]
    # 手势计时器（确保手势维持5秒才发送指令）
    gesture_timers = {
        "gesture_one": 0,
        "gesture_two": 0,
        "gesture_three": 0
    }

    print("手势检测进程启动，开始监听摄像头...")

    try:
        while True:
            current_time = time.time()
            # 获取摄像头画面
            image = get_video_frame(client)
            if image is None:
                time.sleep(0.1)
                continue

            # 绘制检测区域并创建掩码
            mask = draw_detection_area(image)
            # 转换颜色空间（OpenCV BGR → MediaPipe RGB）
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 仅检测掩码区域内的画面
            masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

            # 执行手势检测
            result = hands.process(masked_img)
            flag = 0  # 记录伸直的手指数量

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    # 绘制手部关键点和骨骼连接
                    drawing_utils.draw_landmarks(
                        image, hand_lms, mp_hands.HAND_CONNECTIONS,
                        drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                        drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )

                    # 获取关键节点坐标（0:手腕, 4:大拇指指尖, 5:食指根, 8:食指尖, 12:中指尖, 16:无名指尖, 20:小指尖）
                    landmarks = hand_lms.landmark
                    p0 = (landmarks[0].x, landmarks[0].y)  # 手腕
                    p5 = (landmarks[5].x, landmarks[5].y)  # 食指根
                    p4 = (landmarks[4].x, landmarks[4].y)  # 大拇指指尖
                    p8 = (landmarks[8].x, landmarks[8].y)  # 食指尖
                    p12 = (landmarks[12].x, landmarks[12].y)  # 中指尖
                    p16 = (landmarks[16].x, landmarks[16].y)  # 无名指尖
                    p20 = (landmarks[20].x, landmarks[20].y)  # 小指尖

                    # 计算基准距离（手腕到食指根的距离，用于归一化）
                    distance_0_5 = (p0[0]-p5[0])**2 + (p0[1]-p5[1])**2
                    base = distance_0_5 / 0.6  # 基准阈值

                    # 判断各手指是否伸直（通过指尖到手腕的距离判断）
                    # 大拇指（特殊处理：与食指根的距离）
                    distance_5_4 = (p5[0]-p4[0])**2 + (p5[1]-p4[1])**2
                    if distance_5_4 > base * 0.3:
                        flag += 1
                    # 食指
                    distance_0_8 = (p0[0]-p8[0])**2 + (p0[1]-p8[1])**2
                    if distance_0_8 > base:
                        flag += 1
                    # 中指
                    distance_0_12 = (p0[0]-p12[0])**2 + (p0[1]-p12[1])**2
                    if distance_0_12 > base:
                        flag += 1
                    # 无名指
                    distance_0_16 = (p0[0]-p16[0])**2 + (p0[1]-p16[1])**2
                    if distance_0_16 > base:
                        flag += 1
                    # 小指
                    distance_0_20 = (p0[0]-p20[0])**2 + (p0[1]-p20[1])**2
                    if distance_0_20 > base:
                        flag += 1

                    # 限制最大手指数量为10
                    flag = min(flag, 10)

            # 显示当前检测到的手指数量
            cv2.putText(image, f"Detected: {gesture_labels[flag]}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 手势判断逻辑（维持5秒才发送指令，避免误触发）
            if flag == 1 and gesture_last_sent != "gesture_one_detected":
                if gesture_timers["gesture_one"] == 0:
                    gesture_timers["gesture_one"] = current_time
                elif current_time - gesture_timers["gesture_one"] > 5:
                    print("检测到手势：1指 → 发送坐下指令")
                    pipe.send("gesture_one_detected")
                    gesture_last_sent = "gesture_one_detected"
                    gesture_timers["gesture_one"] = 0
                    last_action_time_shared.value = current_time
            elif flag != 1:
                gesture_timers["gesture_one"] = 0

            if flag == 2 and gesture_last_sent != "gesture_two_detected":
                if gesture_timers["gesture_two"] == 0:
                    gesture_timers["gesture_two"] = current_time
                elif current_time - gesture_timers["gesture_two"] > 5:
                    print("检测到手势：2指 → 发送打招呼指令")
                    pipe.send("gesture_two_detected")
                    gesture_last_sent = "gesture_two_detected"
                    gesture_timers["gesture_two"] = 0
                    last_action_time_shared.value = current_time
            elif flag != 2:
                gesture_timers["gesture_two"] = 0

            if flag == 3 and gesture_last_sent != "gesture_three_detected":
                if gesture_timers["gesture_three"] == 0:
                    gesture_timers["gesture_three"] = current_time
                elif current_time - gesture_timers["gesture_three"] > 5:
                    print("检测到手势：3指 → 发送比爱心指令")
                    pipe.send("gesture_three_detected")
                    gesture_last_sent = "gesture_three_detected"
                    gesture_timers["gesture_three"] = 0
                    last_action_time_shared.value = current_time
            elif flag != 3:
                gesture_timers["gesture_three"] = 0

            # 无手势时发送站立指令（维持机器人站立状态）
            if flag == 0 and gesture_last_sent != "stand":
                pipe.send("stand")
                gesture_last_sent = "stand"

            # 显示摄像头画面（退出按 ESC 键）
            cv2.imshow("Go2 Gesture Control", image)
            if cv2.waitKey(1) == 27:  # ESC 键退出
                print("收到退出信号，关闭手势检测进程")
                break

    except Exception as e:
        print(f"手势检测进程异常: {e}")
    finally:
        # 资源释放
        hands.close()
        cv2.destroyAllWindows()
        client.Close()
        pipe.close()

# 机器人控制进程（独立线程处理手势指令）
def robot_control_process(pipe, network_interface, last_action_time_shared):
    try:
        # 初始化通信通道和运动控制客户端
        ChannelFactoryInitialize(0, network_interface)
        robot = SportModeTest()
        dog_is_standing = True
        print("机器人控制进程启动，等待手势指令...")

        while True:
            try:
                # 接收手势指令（阻塞等待）
                gesture = pipe.recv()
                print(f"收到手势指令: {gesture}")

                # 执行对应动作
                if gesture == "gesture_one_detected":
                    robot.Sit()
                    dog_is_standing = False
                elif gesture == "gesture_two_detected":
                    # 确保先站立再做打招呼动作
                    if not dog_is_standing:
                        robot.Stand()
                        time.sleep(1.5)  # 等待站立完成
                    robot.Hello()
                    dog_is_standing = True
                elif gesture == "gesture_three_detected":
                    # 确保先站立再做比爱心动作
                    if not dog_is_standing:
                        robot.Stand()
                        time.sleep(1.5)  # 等待站立完成
                    robot.Heart()
                    dog_is_standing = True
                elif gesture == "stand":
                    if not dog_is_standing:
                        robot.Stand()
                        dog_is_standing = True

                # 动作执行后等待1秒，避免指令冲突
                time.sleep(1.0)

            except EOFError:
                print("通信管道关闭，退出控制进程")
                break
            except Exception as e:
                print(f"机器人动作执行异常: {e}")
                # 异常后尝试恢复站立状态
                robot.Stand()
                dog_is_standing = True
                time.sleep(2.0)

    except Exception as e:
        print(f"机器人控制进程初始化失败: {e}")
    finally:
        pipe.close()

# 信号处理函数（优雅退出进程）
def signal_handler(sig, frame):
    print("\n收到退出信号，正在终止所有进程...")
    sys.exit(0)

# 主函数（进程创建和管理）
def main():
    # 注册信号处理（支持 Ctrl+C 退出）
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 检查命令行参数（需要传入网络接口，如 wlo1）
    if len(sys.argv) != 2:
        print("用法: python3 gesture_control_go2.py <网络接口>")
        print("示例: python3 gesture_control_go2.py wlo1")
        sys.exit(1)
    network_interface = sys.argv[1]

    # 创建进程间通信管道
    parent_conn, child_conn = mp.Pipe()
    # 创建共享变量（记录最后动作时间）
    last_action_time_shared = Value('d', time.time())

    # 创建并启动进程
    p1 = mp.Process(target=gesture_detection_process, args=(parent_conn, network_interface, last_action_time_shared))
    p2 = mp.Process(target=robot_control_process, args=(child_conn, network_interface, last_action_time_shared))

    p1.start()
    p2.start()

    try:
        # 等待进程结束
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("键盘中断，终止所有进程")
    finally:
        # 确保进程终止
        if p1.is_alive():
            p1.terminate()
            p1.join()
        if p2.is_alive():
            p2.terminate()
            p2.join()
        print("所有进程已终止")

if __name__ == "__main__":
    main()
