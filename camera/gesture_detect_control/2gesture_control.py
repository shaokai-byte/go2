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

# ==================== 配置常量 ====================
class Config:
    GESTURE_HOLD_TIME = 1.5      # 手势维持时间（1.5秒）
    ACTION_COOLDOWN = 2.0        # 动作冷却时间（秒）
    RECONNECT_MAX_ATTEMPTS = 10  # 视频重连最大尝试次数
    DEBUG_MODE = False           # 调试模式（显示更多信息）
    DETECTION_RADIUS = 400       # 手势检测区域半径（扩大一倍，原来是200）
    CONFIDENCE_THRESHOLD = 0.6   # 手势检测置信度阈值

# ==================== 机器人运动控制类 ====================
class SportModeTest:
    def __init__(self) -> None:
        self.client = None
        self.initialized = False
        self.max_retries = 3
        self._init_client()
    
    def _init_client(self):
        """初始化运动控制客户端（带重试机制）"""
        for attempt in range(self.max_retries):
            try:
                self.client = SportClient()
                self.client.SetTimeout(10.0)
                self.client.Init()
                self.initialized = True
                print("SportClient 初始化成功")
                return True
            except Exception as e:
                print(f"SportClient 初始化失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                time.sleep(1)
        return False

    def Stand(self):
        if not self.initialized:
            print("运动控制客户端未初始化")
            return False
        try:
            self.client.StandUp()
            print("机器人执行：站立")
            return True
        except Exception as e:
            print(f"站立动作执行失败: {e}")
            return False

    def Sit(self):
        if not self.initialized:
            print("运动控制客户端未初始化")
            return False
        try:
            self.client.Sit()
            print("机器人执行：坐下")
            return True
        except Exception as e:
            print(f"坐下动作执行失败: {e}")
            return False

    def Hello(self):
        if not self.initialized:
            print("运动控制客户端未初始化")
            return False
        try:
            self.client.Hello()
            print("机器人执行：打招呼")
            return True
        except Exception as e:
            print(f"打招呼动作执行失败: {e}")
            return False

    def Heart(self):
        if not self.initialized:
            print("运动控制客户端未初始化")
            return False
        try:
            self.client.Heart()
            print("机器人执行：比爱心")
            return True
        except Exception as e:
            print(f"比爱心动作执行失败: {e}")
            return False

# ==================== 视频客户端初始化 ====================
def initialize_video_client(network_interface):
    """初始化视频客户端"""
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

# ==================== 获取视频帧 ====================
def get_video_frame(client):
    """获取并解码视频帧"""
    if client is None:
        return None
    try:
        code, data = client.GetImageSample()
        if code == 0:
            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            return image
        else:
            if Config.DEBUG_MODE:
                print(f"获取视频帧失败，错误码: {code}")
            return None
    except Exception as e:
        if Config.DEBUG_MODE:
            print(f"视频帧解码异常: {e}")
        return None

# ==================== 绘制手势检测区域 ====================
def draw_detection_area(image, radius=None):
    """绘制圆形手势检测区域并返回掩码"""
    if radius is None:
        radius = Config.DETECTION_RADIUS
    
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    
    # 绘制检测区域圆（扩大后的窗口）
    cv2.circle(image, center, radius, (0, 255, 0), 4)
    cv2.putText(image, "Gesture Detection Area (2x Expanded)", (center[0]-180, center[1]-radius-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制中心点
    cv2.circle(image, center, 8, (0, 0, 255), -1)
    
    # 添加半径标注
    cv2.putText(image, f"Radius: {radius}px", (center[0]+20, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # 创建掩码（仅检测圆形区域内的手势）
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    
    return mask, center

# ==================== 改进的手指计数函数 ====================
def calculate_angle(p1, p2, p3):
    """计算三点之间的角度（度数）"""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    
    ba = a - b
    bc = c - b
    
    # 避免除零错误
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    
    if ba_norm == 0 or bc_norm == 0:
        return 180
    
    cos_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
    # 限制范围避免数值错误
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def detect_finger_count(hand_landmarks, image_shape=None):
    """
    准确检测伸直的手指数量
    使用角度检测方法，更准确可靠
    """
    landmarks = hand_landmarks.landmark
    count = 0
    
    # 手指关键点索引（指尖, 第二关节, 第三关节）
    finger_tips = [4, 8, 12, 16, 20]    # 指尖
    finger_pips = [3, 6, 10, 14, 18]    # 第二关节
    finger_mcps = [2, 5, 9, 13, 17]     # 第三关节/掌指关节
    
    # 大拇指特殊处理
    thumb_angle = calculate_angle(
        landmarks[finger_pips[0]], 
        landmarks[finger_mcps[0]], 
        landmarks[finger_tips[0]]
    )
    
    # 根据手部朝向判断
    if thumb_angle > 150:
        count += 1
    elif image_shape is not None:
        if landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x:
            count += 1
    
    # 检查其他四指
    for i in range(1, 5):
        tip_idx = finger_tips[i]
        pip_idx = finger_pips[i]
        mcp_idx = finger_mcps[i]
        
        angle = calculate_angle(landmarks[tip_idx], landmarks[pip_idx], landmarks[mcp_idx])
        
        if angle >= 160:
            count += 1
    
    return count

# ==================== 手势检测进程 ====================
def gesture_detection_process(pipe, network_interface, last_action_time_shared, action_queue_shared):
    """独立进程处理手势检测"""
    
    # 初始化视频客户端（带重试机制）
    client = None
    reconnect_attempts = 0
    
    print("正在初始化视频客户端...")
    
    while client is None and reconnect_attempts < Config.RECONNECT_MAX_ATTEMPTS:
        client = initialize_video_client(network_interface)
        if client is None:
            reconnect_attempts += 1
            print(f"视频初始化失败，等待重试 ({reconnect_attempts}/{Config.RECONNECT_MAX_ATTEMPTS})...")
            time.sleep(2)
    
    if client is None:
        print("视频客户端初始化失败，退出手势检测进程")
        pipe.close()
        return

    # 初始化 MediaPipe 手势检测
    mp_hands = mp_mediapipe.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=Config.CONFIDENCE_THRESHOLD,
        min_tracking_confidence=Config.CONFIDENCE_THRESHOLD
    )
    
    # 手势状态变量
    gesture_last_sent = None
    gesture_labels = ["✋ 无", "☝️ 1指", "✌️ 2指", "🤟 3指", "🖖 4指", "🖐️ 5指", 
                      "6指", "7指", "8指", "9指", "10指"]
    
    # 手势计时器
    gesture_timers = {
        1: 0,   # 1指 - 坐下
        2: 0,   # 2指 - 打招呼
        3: 0,   # 3指 - 比爱心
    }
    
    # 帧率控制
    frame_count = 0
    last_fps_time = time.time()
    fps = 0
    
    # 缓存掩码（性能优化）
    cached_mask = None
    cached_center = None
    last_image_shape = None
    
    print("手势检测进程启动，开始监听摄像头...")
    print("=" * 60)
    print("控制说明:")
    print(f"  - 伸出 1 根手指并保持 {Config.GESTURE_HOLD_TIME} 秒 → 机器人坐下")
    print(f"  - 伸出 2 根手指并保持 {Config.GESTURE_HOLD_TIME} 秒 → 机器人打招呼")
    print(f"  - 伸出 3 根手指并保持 {Config.GESTURE_HOLD_TIME} 秒 → 机器人比爱心")
    print("  - 握拳（0 指） → 机器人站立（立即执行）")
    print(f"  - 检测区域半径: {Config.DETECTION_RADIUS} 像素 (已扩大一倍)")
    print("  - 按 ESC 键退出程序")
    print("  - 按 D 键切换调试模式")
    print("=" * 60)
    
    try:
        while True:
            current_time = time.time()
            
            # 计算帧率
            frame_count += 1
            if current_time - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = current_time
            
            # 获取摄像头画面
            image = get_video_frame(client)
            if image is None:
                print("视频帧获取失败，尝试重新连接...")
                if client:
                    client.Close()
                client = initialize_video_client(network_interface)
                if client is None:
                    time.sleep(1)
                continue
            
            # 获取图像尺寸
            height, width, _ = image.shape
            current_shape = (height, width)
            
            # 性能优化：只在尺寸变化时重新创建掩码
            if cached_mask is None or last_image_shape != current_shape:
                cached_mask, cached_center = draw_detection_area(image.copy(), Config.DETECTION_RADIUS)
                last_image_shape = current_shape
            
            # 应用掩码（只检测圆形区域）
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=cached_mask)
            
            # 执行手势检测
            result = hands.process(masked_img)
            flag = 0  # 伸直手指数量
            
            # 在图像上绘制检测区域
            cv2.circle(image, cached_center, Config.DETECTION_RADIUS, (0, 255, 0), 4)
            cv2.putText(image, "Gesture Detection Area (2x Expanded)", 
                       (cached_center[0]-180, cached_center[1]-Config.DETECTION_RADIUS-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.circle(image, cached_center, 8, (0, 0, 255), -1)
            
            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    # 只在调试模式下绘制关键点
                    if Config.DEBUG_MODE:
                        mp_mediapipe.solutions.drawing_utils.draw_landmarks(
                            image, hand_lms, mp_hands.HAND_CONNECTIONS,
                            mp_mediapipe.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                            mp_mediapipe.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=1)
                        )
                    
                    # 使用改进的手指计数函数
                    flag = detect_finger_count(hand_lms, image.shape)
                    
                    # 可选：显示每个手指的角度（调试用）
                    if Config.DEBUG_MODE:
                        cv2.putText(image, f"Fingers: {flag}", (50, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # 显示检测结果
            gesture_text = f"Detected: {gesture_labels[flag] if flag < len(gesture_labels) else f'{flag}指'}"
            cv2.putText(image, gesture_text, (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # 显示FPS
            cv2.putText(image, f"FPS: {fps}", (width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示操作提示
            cv2.putText(image, "ESC: Exit | D: Debug", (width - 200, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 显示手势维持时间要求
            hold_time_text = f"Hold time required: {Config.GESTURE_HOLD_TIME}s"
            cv2.putText(image, hold_time_text, (50, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # 手势判断逻辑（维持1.5秒才发送指令）
            action_to_send = None
            
            # 1指：坐下
            if flag == 1:
                if gesture_timers[1] == 0:
                    gesture_timers[1] = current_time
                    cv2.putText(image, f"Holding... ({Config.GESTURE_HOLD_TIME}s needed)", (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif current_time - gesture_timers[1] >= Config.GESTURE_HOLD_TIME:
                    if gesture_last_sent != "gesture_one_detected":
                        action_to_send = "gesture_one_detected"
                        print(f"✅ 检测到手势：1指（维持{Config.GESTURE_HOLD_TIME}秒） → 发送坐下指令")
            else:
                gesture_timers[1] = 0
            
            # 2指：打招呼
            if flag == 2:
                if gesture_timers[2] == 0:
                    gesture_timers[2] = current_time
                    cv2.putText(image, f"Holding... ({Config.GESTURE_HOLD_TIME}s needed)", (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif current_time - gesture_timers[2] >= Config.GESTURE_HOLD_TIME:
                    if gesture_last_sent != "gesture_two_detected":
                        action_to_send = "gesture_two_detected"
                        print(f"✅ 检测到手势：2指（维持{Config.GESTURE_HOLD_TIME}秒） → 发送打招呼指令")
            else:
                gesture_timers[2] = 0
            
            # 3指：比爱心
            if flag == 3:
                if gesture_timers[3] == 0:
                    gesture_timers[3] = current_time
                    cv2.putText(image, f"Holding... ({Config.GESTURE_HOLD_TIME}s needed)", (50, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif current_time - gesture_timers[3] >= Config.GESTURE_HOLD_TIME:
                    if gesture_last_sent != "gesture_three_detected":
                        action_to_send = "gesture_three_detected"
                        print(f"✅ 检测到手势：3指（维持{Config.GESTURE_HOLD_TIME}秒） → 发送比爱心指令")
            else:
                gesture_timers[3] = 0
            
            # 握拳（0指）：站立（立即执行，无需等待）
            if flag == 0 and gesture_last_sent != "stand":
                action_to_send = "stand"
                print(f"🦿 检测到握拳 → 发送站立指令")
            
            # 发送动作指令（带冷却时间检查）
            if action_to_send:
                current_action_time = time.time()
                last_action_time = last_action_time_shared.value
                
                # 检查冷却时间
                if current_action_time - last_action_time >= Config.ACTION_COOLDOWN:
                    pipe.send(action_to_send)
                    gesture_last_sent = action_to_send
                    last_action_time_shared.value = current_action_time
                else:
                    if Config.DEBUG_MODE:
                        remaining = Config.ACTION_COOLDOWN - (current_action_time - last_action_time)
                        print(f"⏱️ 动作冷却中，剩余 {remaining:.1f} 秒")
            
            # 显示计时进度条（如果有手势正在计时）
            for gesture_id, start_time in gesture_timers.items():
                if start_time > 0:
                    elapsed = current_time - start_time
                    if elapsed < Config.GESTURE_HOLD_TIME:
                        progress = int((elapsed / Config.GESTURE_HOLD_TIME) * 100)
                        bar_length = 200
                        filled = int(bar_length * elapsed / Config.GESTURE_HOLD_TIME)
                        
                        # 进度条背景
                        cv2.rectangle(image, (50, 160), (50 + bar_length, 180), (100, 100, 100), -1)
                        # 进度条前景
                        cv2.rectangle(image, (50, 160), (50 + filled, 180), (0, 255, 0), -1)
                        
                        # 显示百分比
                        cv2.putText(image, f"{progress}%", (50 + bar_length + 10, 175),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示当前机器人状态
            cv2.putText(image, "Robot: Stand", (50, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            
            # 显示检测区域半径信息
            cv2.putText(image, f"Detection Zone: {Config.DETECTION_RADIUS}px (2x)", 
                       (cached_center[0] - 150, cached_center[1] + Config.DETECTION_RADIUS + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 显示摄像头画面
            cv2.imshow("Go2 Gesture Control - Detection Zone 2x", image)
            
            # 按 ESC 键退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n收到 ESC 退出信号，关闭手势检测进程")
                break
            elif key == ord('d') or key == ord('D'):
                # 按 D 键切换调试模式
                Config.DEBUG_MODE = not Config.DEBUG_MODE
                print(f"调试模式: {'开启' if Config.DEBUG_MODE else '关闭'}")
    
    except Exception as e:
        print(f"手势检测进程异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 资源释放
        hands.close()
        cv2.destroyAllWindows()
        if client:
            client.Close()
        pipe.close()
        print("手势检测进程已关闭")

# ==================== 机器人控制进程 ====================
def robot_control_process(pipe, network_interface, last_action_time_shared):
    """独立进程处理机器人控制"""
    
    # 初始化机器人客户端
    retry_count = 0
    robot = None
    
    while robot is None and retry_count < 3:
        try:
            ChannelFactoryInitialize(0, network_interface)
            robot = SportModeTest()
            if not robot.initialized:
                robot = None
                raise Exception("机器人初始化失败")
        except Exception as e:
            print(f"机器人控制进程初始化失败 (尝试 {retry_count+1}/3): {e}")
            retry_count += 1
            time.sleep(2)
    
    if robot is None:
        print("机器人控制进程初始化失败，退出")
        pipe.close()
        return
    
    dog_is_standing = True
    last_action = None
    action_count = 0
    
    print("🤖 机器人控制进程启动，等待手势指令...")
    print("-" * 50)
    
    # 初始站立
    robot.Stand()
    time.sleep(1)
    
    while True:
        try:
            # 接收手势指令（阻塞等待）
            if not pipe.poll(1.0):  # 1秒超时
                continue
            
            gesture = pipe.recv()
            
            # 避免重复执行相同动作
            if last_action == gesture:
                action_count += 1
                if action_count > 5:  # 连续5次相同指令忽略
                    continue
            else:
                action_count = 0
                last_action = gesture
            
            print(f"\n📨 收到手势指令: {gesture}")
            
            # 执行对应动作
            if gesture == "gesture_one_detected":
                print("🪑 执行：坐下")
                robot.Sit()
                dog_is_standing = False
                
            elif gesture == "gesture_two_detected":
                print("👋 执行：打招呼")
                if not dog_is_standing:
                    print("  先站立...")
                    robot.Stand()
                    time.sleep(1.5)
                robot.Hello()
                dog_is_standing = True
                
            elif gesture == "gesture_three_detected":
                print("❤️ 执行：比爱心")
                if not dog_is_standing:
                    print("  先站立...")
                    robot.Stand()
                    time.sleep(1.5)
                robot.Heart()
                dog_is_standing = True
                
            elif gesture == "stand":
                if not dog_is_standing:
                    print("🧍 执行：站立")
                    robot.Stand()
                    dog_is_standing = True
                else:
                    print("🧍 机器人已在站立状态")
            
            # 动作执行后等待，避免指令冲突
            time.sleep(0.5)
            
        except EOFError:
            print("通信管道关闭，退出控制进程")
            break
        except Exception as e:
            print(f"❌ 机器人动作执行异常: {e}")
            # 异常后尝试恢复站立状态
            try:
                robot.Stand()
                dog_is_standing = True
            except:
                pass
            time.sleep(1.0)
    
    pipe.close()
    print("机器人控制进程已关闭")

# ==================== 信号处理 ====================
def signal_handler(sig, frame):
    """优雅退出处理"""
    print("\n\n⚠️ 收到退出信号，正在终止所有进程...")
    sys.exit(0)

# ==================== 主函数 ====================
def main():
    """主函数：进程创建和管理"""
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("=" * 60)
        print("宇树 Go2 机器狗 - 手势控制系统")
        print("=" * 60)
        print("用法: python3 gesture_control_go2.py <网络接口>")
        print("\n示例:")
        print("  python3 gesture_control_go2.py wlo1      # 无线网卡")
        print("  python3 gesture_control_go2.py eth0      # 有线网卡")
        print("\n查看可用网络接口:")
        print("  Linux: ip link show")
        print("  Windows: ipconfig")
        print("=" * 60)
        sys.exit(1)
    
    network_interface = sys.argv[1]
    
    print("=" * 60)
    print("🐕 宇树 Go2 机器狗 - 手势控制系统")
    print("=" * 60)
    print(f"网络接口: {network_interface}")
    print(f"手势维持时间: {Config.GESTURE_HOLD_TIME} 秒")
    print(f"动作冷却时间: {Config.ACTION_COOLDOWN} 秒")
    print(f"检测区域半径: {Config.DETECTION_RADIUS} 像素 (原始200px，已扩大一倍)")
    print("=" * 60)
    
    # 创建进程间通信管道
    parent_conn, child_conn = mp.Pipe()
    
    # 创建共享变量
    last_action_time_shared = Value('d', time.time())
    action_queue_shared = mp.Queue(maxsize=10)
    
    # 创建并启动进程
    p1 = mp.Process(target=gesture_detection_process, 
                    args=(parent_conn, network_interface, last_action_time_shared, action_queue_shared))
    p2 = mp.Process(target=robot_control_process, 
                    args=(child_conn, network_interface, last_action_time_shared))
    
    p1.start()
    p2.start()
    
    print("\n✅ 系统启动成功！")
    print(f"📹 请将手放在摄像头中央的绿色圆形区域内（半径{Config.DETECTION_RADIUS}px，扩大一倍）")
    print(f"🖐️ 伸出 1/2/3 根手指并保持 {Config.GESTURE_HOLD_TIME} 秒来控制机器人")
    print("🤛 握拳立即让机器人站立")
    print("❌ 按 ESC 键或 Ctrl+C 退出程序\n")
    
    try:
        # 等待进程结束
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("\n⚠️ 键盘中断，终止所有进程")
    finally:
        # 确保进程终止
        print("\n🛑 正在清理资源...")
        if p1.is_alive():
            p1.terminate()
            p1.join(timeout=2)
        if p2.is_alive():
            p2.terminate()
            p2.join(timeout=2)
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("✅ 所有进程已终止，程序退出")

if __name__ == "__main__":
    # 设置多进程启动方式
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()