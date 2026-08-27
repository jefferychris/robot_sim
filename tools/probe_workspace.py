"""用 pose_check 探明工作空间边界 —— 只查询 IK,不驱动机械臂。

背景:config 里 APPROACH/GRASP/PLACE 全被压到 0,是为了绕开 out_of_workspace。
但这样 approach 点和抓取点重合,pick 没有垂直下压、place 不抬升就横移,
很可能撞翻已放好的螺母。要定出安全的抬升高度,先得知道真实边界在哪。

用法:  python tools/probe_workspace.py
"""

import logging
import sys

import numpy as np

logging.basicConfig(level=logging.WARNING)

from agents.nut_picker import config as C
from rabo_robocap import LinkerArmA7

# 视觉当前解算出的三个螺母 + 箱子位置(base 系)
TARGETS = {
    "nut_big":    (-0.305, -0.067, 0.284),
    "nut_medium": (-0.314, -0.168, 0.278),
    "nut_small":  (-0.253, -0.090, 0.288),
    "box":        (-0.385,  0.223, 0.289),
}


def main():
    arm = LinkerArmA7(robot_id=C.LEFT_ARM_ID, mode=C.ARM_MODE)

    def ok(x, y, z):
        try:
            r = arm.pose_check(x, y, z, roll=C.GRIPPER_ROLL,
                               pitch=C.GRIPPER_PITCH, yaw=C.GRIPPER_YAW)
            return bool(r[0] if isinstance(r, (tuple, list)) else r)
        except Exception:
            return False

    print("=" * 66)
    print("[1] 每个目标点的可达 Z 区间(步进 1cm)")
    print("=" * 66)
    for name, (x, y, z0) in TARGETS.items():
        zs = [round(z, 3) for z in np.arange(0.10, 0.61, 0.01) if ok(x, y, round(z, 3))]
        if zs:
            print("%12s (x=%.3f y=%.3f)  可达 Z: %.2f ~ %.2f m   视觉 Z=%.3f %s"
                  % (name, x, y, min(zs), max(zs), z0,
                     "✓在区间内" if min(zs) <= z0 <= max(zs) else "✗超出区间"))
            print("%12s   → 该点之上还有 %.0f mm 余量" % ("", (max(zs) - z0) * 1000))
        else:
            print("%12s (x=%.3f y=%.3f)  ✗ 整条 Z 轴都不可达" % (name, x, y))

    print()
    print("=" * 66)
    print("[2] 抬升高度可行性:每个目标能否 approach 到 z+h")
    print("=" * 66)
    print("%12s %s" % ("target", "  ".join("+%dcm" % int(h * 100) for h in
                                           (0.02, 0.05, 0.08, 0.10, 0.15))))
    for name, (x, y, z0) in TARGETS.items():
        marks = ["  ✓  " if ok(x, y, z0 + h) else "  ✗  "
                 for h in (0.02, 0.05, 0.08, 0.10, 0.15)]
        print("%12s %s" % (name, " ".join(marks)))

    print()
    print("=" * 66)
    print("[3] XY 平面扫描 @ Z=0.28(视觉平面),看整体可达范围")
    print("=" * 66)
    xs = np.arange(-0.50, -0.09, 0.05)
    ys = np.arange(-0.30, 0.31, 0.05)
    print("      " + "".join("%6.2f" % y for y in ys))
    for x in xs:
        row = "".join("     %s" % ("#" if ok(round(x, 3), round(y, 3), 0.28) else ".")
                      for y in ys)
        print("%6.2f%s" % (x, row))
    print("\n(# = 可达, . = 不可达; 行=X, 列=Y)")

    try:
        arm.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
