import time
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

def move_robot_by_distance(sport_client, target_distance, speed=0.5, interval=0.1):
    """
    控制机器人前进指定距离（基于速度和时间估算，如需精准需结合里程计）
    :param sport_client: 初始化后的SportClient实例
    :param target_distance: 目标前进距离（米，正数前进，负数后退）
    :param speed: 移动速度（米/秒，建议0.1~0.8之间，避免过快）
    :param interval: 每次发送移动指令的时间间隔（秒，越小越精准）
    :return: 实际移动的距离（估算值）
    """
    # 处理后退逻辑（速度取反）
    actual_speed = speed if target_distance > 0 else -abs(speed)
    target_abs = abs(target_distance)
    
    current_distance = 0.0  # 已移动距离（估算）
    print(f"开始移动：目标距离 {target_distance} 米，移动速度 {actual_speed} 米/秒")

    # 循环发送移动指令直到达到目标距离
    try:
        while current_distance < target_abs:
            # 发送移动指令
            ret = sport_client.Move(actual_speed, 0, 0)
            if ret != 0:
                print(f"警告：移动指令发送失败，返回值 {ret}")
                break
            
            # 等待指定间隔
            time.sleep(interval)
            
            # 累加已移动距离（估算）
            current_distance += abs(actual_speed) * interval
            
            # 打印实时进度（可选）
            if int(current_distance / interval) % 10 == 0:  # 每0.1米打印一次
                print(f"当前已移动：{current_distance:.2f} / {target_abs:.2f} 米")
    except KeyboardInterrupt:
        print("\n用户中断，立即停止移动")
    finally:
        # 无论是否完成，最终都停止机器人
        sport_client.Move(0, 0, 0)
        time.sleep(0.2)  # 确保停止指令生效
        print(f"移动结束，实际估算移动距离：{current_distance:.2f} 米")
    
    # 返回实际移动距离（带方向）
    return current_distance * (1 if target_distance > 0 else -1)

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    
    # 1. 初始化SDK通道
    try:
        if len(sys.argv) > 1:
            ChannelFactoryInitialize(0, sys.argv[1])  # 传入IP参数（如机器人IP）
        else:
            ChannelFactoryInitialize(0)  # 默认本地/自动发现
        print("通道初始化成功")
    except Exception as e:
        print(f"通道初始化失败：{e}")
        sys.exit(1)
    
    # 2. 初始化运动客户端
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)  # 设置超时时间
    try:
        sport_client.Init()
        print("运动客户端初始化成功")
    except Exception as e:
        print(f"运动客户端初始化失败：{e}")
        sys.exit(1)
    
    # 3. 核心逻辑：控制机器人前进指定距离
    target_distance = 5.0  # 目标前进5米（改为负数则后退）
    move_speed = 0.5       # 移动速度0.5m/s（建议不超过0.8，避免机器人不稳）
    actual_move_distance = move_robot_by_distance(
        sport_client=sport_client,
        target_distance=target_distance,
        speed=move_speed,
        interval=0.1
    )
    
    # 4. 结束提示
    print(f"\n任务完成！目标前进 {target_distance} 米，实际估算前进 {actual_move_distance:.2f} 米")
