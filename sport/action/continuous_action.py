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

    # 仅保留id=3对应的前进移动动作逻辑
    print("Starting move forward action...")
    # 执行前进移动序列
    ret = sport_client.Move(0.5, 0, 0)
    time.sleep(1)
    # sport_client.Move(0.5, 0, 0)
    # time.sleep(1)
    # sport_client.FrontPounce()      #扑人
    # time.sleep(1)
    # sport_client.FrontJump()        #前跳
    # time.sleep(1)
    # sport_client.Hello()            #打招呼
    # time.sleep(1)
    # sport_client.Dance1()           #舞蹈1-18秒
    # time.sleep(1)
    # sport_client.Dance2()           #舞蹈2-45秒
    # time.sleep(1)
    # sport_client.HandStand()        #倒立0
    # time.sleep(1)
    # sport_client.Stretch()          #伸展0
    # time.sleep(1)
    sport_client.Sit()                #坐下
    time.sleep(1)
    sport_client.RiseSit()            #坐下到站立
    time.sleep(1)
    print("Move forward action completed, ret: ", ret)
