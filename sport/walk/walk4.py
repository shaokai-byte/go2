import time
import sys
import math
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# 兼容不同SDK版本的StateClient导入方式
try:
    # 方式1：新版SDK路径
    from unitree_sdk2py.go2.sport.state_client import StateClient
except ImportError:
    try:
        # 方式2：旧版SDK路径
        from unitree_sdk2py.go2.sport import StateClient
    except ImportError:
        # 方式3：终极兼容（无StateClient时用速度估算降级）
        StateClient = None
        print("警告：未找到StateClient模块，将使用速度×时间估算距离（精度略低）")

def init_robot_clients():
    """初始化机器人运动客户端和状态客户端（兼容不同SDK版本）"""
    # 初始化通道
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    
    # 初始化运动客户端
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    
    # 初始化状态客户端（兼容无StateClient的情况）
    state_client = None
    if StateClient is not None:
        try:
            state_client = StateClient()
            state_client.Init()
            print("运动客户端+状态客户端初始化成功")
        except Exception as e:
            print(f"状态客户端初始化失败：{e}，将使用速度估算模式")
    else:
        print("仅初始化运动客户端，使用速度估算模式")
    
    return sport_client, state_client

def move_robot_by_distance(sport_client, state_client, target_distance, speed=0.5, interval=0.1):
    """
    控制机器人前进/后退指定距离（优先用里程计，无则用速度估算）
    :param sport_client: 运动客户端实例
    :param state_client: 状态客户端实例（可为None）
    :param target_distance: 目标距离（米，正数前进，负数后退）
    :param speed: 移动速度（米/秒，0.1~0.8）
    :param interval: 指令发送间隔（秒）
    :return: 实际移动距离（米）
    """
    # 处理前进/后退逻辑
    actual_speed = speed if target_distance > 0 else -abs(speed)
    target_abs = abs(target_distance)
    
    # 初始化距离变量
    current_distance = 0.0
    start_time = time.time()
    start_x = None
    
    # 如果有StateClient，读取初始里程计
    if state_client is not None:
        try:
            start_state = state_client.GetState()
            start_x = start_state.odometry.x  # 初始x轴里程（前进方向）
        except Exception as e:
            print(f"读取初始里程计失败：{e}，切换为速度估算模式")
            start_x = None
    
    print(f"\n开始移动：目标{'前进' if target_distance>0 else '后退'} {target_abs} 米，速度 {abs(actual_speed)} m/s")
    
    try:
        while current_distance < target_abs:
            # 发送移动指令
            sport_client.Move(actual_speed, 0, 0)
            time.sleep(interval)
            
            # 计算当前移动距离（优先里程计，无则用速度×时间）
            if start_x is not None and state_client is not None:
                try:
                    current_state = state_client.GetState()
                    current_x = current_state.odometry.x
                    current_distance = abs(current_x - start_x)  # 里程计精准值
                except Exception as e:
                    # 里程计读取失败时降级为速度估算
                    current_distance = abs(actual_speed) * (time.time() - start_time)
                    print(f"读取里程计异常：{e}，临时使用速度估算", end="\r")
            else:
                # 速度估算模式
                current_distance = abs(actual_speed) * (time.time() - start_time)
            
            # 打印实时进度
            print(f"实时移动距离：{current_distance:.2f} / {target_abs:.2f} 米", end="\r")
        
        print(f"\n移动完成，实际移动距离：{current_distance:.2f} 米")
    except KeyboardInterrupt:
        print("\n用户中断，停止移动")
    finally:
        sport_client.Move(0, 0, 0)  # 停止移动
        time.sleep(0.2)
    
    return current_distance

def rotate_robot_by_angle(sport_client, state_client, target_angle, rotate_speed=1.0, interval=0.1):
    """
    控制机器人旋转指定角度（优先用偏航角，无则用时间估算）
    :param sport_client: 运动客户端实例
    :param state_client: 状态客户端实例（可为None）
    :param target_angle: 目标旋转角度（度，正数左转，负数右转）
    :param rotate_speed: 旋转角速度（rad/s，建议0.5~2.0）
    :param interval: 指令发送间隔（秒）
    :return: 实际旋转角度（度）
    """
    # 角度转弧度
    target_rad = math.radians(target_angle)
    target_rad_abs = abs(target_rad)
    actual_rotate_speed = rotate_speed if target_angle > 0 else -abs(rotate_speed)
    
    # 初始化旋转变量
    current_rad = 0.0
    start_time = time.time()
    start_yaw = None
    
    # 如果有StateClient，读取初始偏航角
    if state_client is not None:
        try:
            start_state = state_client.GetState()
            start_yaw = start_state.odometry.yaw  # 初始偏航角（弧度）
        except Exception as e:
            print(f"读取初始偏航角失败：{e}，切换为时间估算模式")
            start_yaw = None
    
    print(f"\n开始旋转：目标{'左转' if target_angle>0 else '右转'} {abs(target_angle)}°，角速度 {abs(rotate_speed)} rad/s")
    
    try:
        while current_rad < target_rad_abs:
            # 发送旋转指令（vx=0, vy=0, wz=旋转角速度）
            sport_client.Move(0, 0, actual_rotate_speed)
            time.sleep(interval)
            
            # 计算当前旋转角度（优先偏航角，无则用时间估算）
            if start_yaw is not None and state_client is not None:
                try:
                    current_state = state_client.GetState()
                    current_yaw = current_state.odometry.yaw
                    # 处理偏航角跨0度（360度）的情况
                    delta_yaw = math.fmod(current_yaw - start_yaw, 2 * math.pi)
                    current_rad = abs(delta_yaw)
                except Exception as e:
                    # 偏航角读取失败时降级为时间估算
                    current_rad = abs(actual_rotate_speed) * (time.time() - start_time)
                    print(f"读取偏航角异常：{e}，临时使用时间估算", end="\r")
            else:
                # 时间估算模式（角速度×时间）
                current_rad = abs(actual_rotate_speed) * (time.time() - start_time)
            
            # 打印实时进度
            current_angle = math.degrees(current_rad)
            print(f"实时旋转角度：{current_angle:.1f} / {abs(target_angle):.1f}°", end="\r")
        
        print(f"\n旋转完成，实际旋转角度：{math.degrees(current_rad):.1f}°")
    except KeyboardInterrupt:
        print("\n用户中断，停止旋转")
    finally:
        sport_client.Move(0, 0, 0)  # 停止旋转
        time.sleep(0.2)
    
    return math.degrees(current_rad)

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    
    # 1. 初始化客户端
    sport_client, state_client = init_robot_clients()
    
    # 2. 执行移动+旋转序列（可自定义修改）
    try:
        # 第一步：前进4米
        move_robot_by_distance(
            sport_client=sport_client,
            state_client=state_client,
            target_distance=4.0,  # 前进4米（负数为后退）
            speed=0.5,
            interval=0.05
        )
        
        # 第二步：左转90度
        rotate_robot_by_angle(
            sport_client=sport_client,
            state_client=state_client,
            target_angle=90,  # 左转90度（负数为右转）
            rotate_speed=1.0  # 旋转角速度1rad/s
        )
        
        # 第三步：再前进2.0米
        move_robot_by_distance(
            sport_client=sport_client,
            state_client=state_client,
            target_distance=2.0,
            speed=0.5,
            interval=0.05
        )
        
        # 第四步：左转90度
        rotate_robot_by_angle(
            sport_client=sport_client,
            state_client=state_client,
            target_angle=90,  # 左转90度
            rotate_speed=0.8
        )
        
        # 第五步：前进4米
        move_robot_by_distance(
            sport_client=sport_client,
            state_client=state_client,
            target_distance=4.0,  # 前进4米
            speed=0.5,
            interval=0.05
        )
        rotate_robot_by_angle(
            sport_client=sport_client,
            state_client=state_client,
            target_angle=90,  # 左转90度
            rotate_speed=0.8
        )
        move_robot_by_distance(
            sport_client=sport_client,
            state_client=state_client,
            target_distance=2.0,  # 前进2米
            speed=0.5,
            interval=0.05
        )
        
    except Exception as e:
        print(f"\n执行过程中出错：{e}")
        sport_client.Move(0, 0, 0)  # 异常时停止机器人
    finally:
        # 最终确保机器人停止
        sport_client.Move(0, 0, 0)
        print("\n所有动作执行完成，机器人已停止")
