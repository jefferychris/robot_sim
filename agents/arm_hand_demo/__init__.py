"""左臂向前伸直 + 左手竖大拇指 演示。"""

import time
from rabo_robocap import LinkerArmA7, LinkerHandO6Left

# 场景中的机器人 ID
LEFT_ARM_ID = "rbd03ebf4ebf83c6a6a64754454bc520a"
LEFT_HAND_ID = "r136d7b4b6e527ea3875679b4bf7eeb7d"


def run():
    # 初始化左臂和左手
    arm = LinkerArmA7(robot_id=LEFT_ARM_ID, mode="sim")
    hand = LinkerHandO6Left(robot_id=LEFT_HAND_ID, mode="sim")

    # 1) 左臂向前伸直: J1=-π/2(上臂前倾), J4=π/2(肘伸直向前)
    print("左臂向前伸直...")
    arm.move_joints([0, -3.14/2, 0.0, 0, 0.0, 0.0, 0.0])
    arm.move_joints([-3.14/2, -3.14/2, 0.0, 0, 0.0, 0.0, 0.0])
    arm.move_joints([-3.14/2, -3.14/2, -3.14/2, 0, 0.0, 0.0, 0.0])

    time.sleep(2.0)

    # 2) 左手竖大拇指: 拇指竖起(thumb_bend=0)，四指握拳(=1)
    print("左手竖大拇指...")
    hand.clench(thumb_rotation=0, thumb_bend=0, index=1, middle=1, ring=1, pinky=1)

    print("完成!")
    arm.shutdown()
    hand.shutdown()
