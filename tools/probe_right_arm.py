"""验证右臂能否覆盖三个螺母 —— 这是 pick 全线 out_of_workspace 的根因验证。

发现:tools.probe_workspace 的 XY 扫描显示 Y < -0.05 整片不可达,而三个螺母
的 Y 是 -0.067 / -0.168 / -0.090,全在禁区;箱子 Y=+0.223 可达且余量 141mm。
config 用的是 LEFT_ARM(注释"只动左臂"),但官方数据集 README 写明本任务是
"Single right-hand operation" —— 螺母摆在机器人右侧,左臂够不着是几何必然。

本脚本用右臂重跑同样的可达性检查。不驱动机械臂,只查 IK。

用法:  python -m tools.probe_right_arm
"""

import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)

from agents.nut_picker import config as C
from rabo_robocap import LinkerArmA7

RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"   # 由 camera topic frame_id 反推

TARGETS = {
    "nut_big":    (-0.305, -0.067, 0.284),
    "nut_medium": (-0.314, -0.168, 0.278),
    "nut_small":  (-0.253, -0.090, 0.288),
    "box":        (-0.305,  0.260, 0.298),
}


def check(arm, x, y, z, pitch):
    try:
        r = arm.pose_check(x, y, z, roll=C.GRIPPER_ROLL, pitch=pitch, yaw=C.GRIPPER_YAW)
        return bool(r[0] if isinstance(r, (tuple, list)) else r)
    except Exception:
        return False


def main():
    arm = LinkerArmA7(robot_id=RIGHT_ARM_ID, mode=C.ARM_MODE)
    print("右臂 ID:", RIGHT_ARM_ID)

    print("\n" + "=" * 66)
    print("[1] 右臂:每个目标的可达 Z 区间(俯视姿态 pitch=π)")
    print("=" * 66)
    for name, (x, y, z0) in TARGETS.items():
        zs = [round(z, 3) for z in np.arange(0.10, 0.61, 0.01)
              if check(arm, x, y, round(z, 3), C.GRIPPER_PITCH)]
        if zs:
            head = (max(zs) - z0) * 1000
            print("%12s  可达 Z: %.2f ~ %.2f m   视觉 Z=%.3f  上方余量 %.0f mm %s"
                  % (name, min(zs), max(zs), z0, head,
                     "✓" if min(zs) <= z0 <= max(zs) else "✗超出"))
        else:
            print("%12s  ✗ 整条 Z 轴不可达" % name)

    print("\n" + "=" * 66)
    print("[2] 右臂:抬升高度可行性")
    print("=" * 66)
    hs = (0.02, 0.05, 0.08, 0.10, 0.15)
    print("%12s %s" % ("target", "  ".join("+%dcm" % int(h * 100) for h in hs)))
    for name, (x, y, z0) in TARGETS.items():
        print("%12s %s" % (name, " ".join(
            "  ✓  " if check(arm, x, y, z0 + h, C.GRIPPER_PITCH) else "  ✗  " for h in hs)))

    print("\n" + "=" * 66)
    print("[3] 右臂:XY 平面扫描 @ Z=0.28")
    print("=" * 66)
    ys = np.arange(-0.30, 0.31, 0.05)
    print("      " + "".join("%6.2f" % y for y in ys))
    for x in np.arange(-0.50, -0.09, 0.05):
        print("%6.2f%s" % (x, "".join(
            "     %s" % ("#" if check(arm, round(x, 3), round(y, 3), 0.28,
                                      C.GRIPPER_PITCH) else ".") for y in ys)))
    print("\n(# = 可达, . = 不可达; 行=X, 列=Y)")

    print("\n" + "=" * 66)
    print("[4] 若俯视仍受限:换倾斜姿态,螺母 @ 视觉 Z + 5cm")
    print("=" * 66)
    pitches = [("π (俯视)", np.pi), ("2.8", 2.8), ("2.6", 2.6), ("2.4", 2.4), ("2.2", 2.2)]
    print("%12s %s" % ("target", "  ".join("%9s" % p[0] for p in pitches)))
    for name, (x, y, z0) in TARGETS.items():
        print("%12s %s" % (name, "  ".join(
            "%9s" % ("✓" if check(arm, x, y, z0 + 0.05, p) else "✗") for _, p in pitches)))

    try:
        arm.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
