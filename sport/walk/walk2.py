import time
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")
    
    # 初始化通道
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # 初始化运动客户端
    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    print("Starting move forward action...")
    # 定义移动序列：[(vx, vy, wz, duration), ...]
    move_sequence = [
        (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1),
        (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1),
        (0.3, 0, 0, 1), (0, 0, 1.5, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1),
        (0.5, 0, 0, 1), (0.3, 0, 0, 1), (0, 0, 1.5, 1), (0.5, 0, 0, 1),
        (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1),
        (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.3, 0, 0, 1),
        (0, 0, 1.5, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1), (0.5, 0, 0, 1)
    ]
    
    # 循环执行移动序列
    ret = None
    for vx, vy, wz, duration in move_sequence:
        ret = sport_client.Move(vx, vy, wz)
        time.sleep(duration)

    print("Move forward action completed, ret: ", ret)
