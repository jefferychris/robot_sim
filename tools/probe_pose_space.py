"""扫描姿态空间:同一目标点,换不同 roll/pitch/yaw 找可达解。

前两轮排除的结论:
  - 换手臂无效(左右臂可达区几乎相同)
  - 整体平移到"world→base"无效(平移后连箱子都不可达,比原来更差)
  - 原坐标下箱子可达且余量 202mm → 坐标数量级本身是对的

剩下的解释:一直在用固定姿态 (roll=0, pitch=π, yaw=0) 查 IK,那只是 7 自由度
解空间里的一个切片。Y 负半区在该姿态下无解,不代表换个手腕朝向也无解。

本脚本对每个目标点扫 roll×pitch×yaw 网格,找出任一可行姿态。

用法:  python -m tools.probe_pose_space
"""

import itertools
import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)

from agents.nut_picker import config as C
from rabo_robocap import LinkerArmA7

RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"

TARGETS = {
    "nut_big":    (-0.305, -0.067, 0.284),
    "nut_medium": (-0.314, -0.168, 0.278),
    "nut_small":  (-0.253, -0.090, 0.288),
    "box":        (-0.305,  0.260, 0.298),
}

ROLLS   = [0.0, 0.5, 1.0, -0.5, -1.0, 1.57, -1.57]
PITCHES = [np.pi, 2.9, 2.7, 2.5, 2.2, 1.9, 1.57, 1.2]
YAWS    = [0.0, 0.5, 1.0, 1.57, -0.5, -1.0, -1.57, 2.5, 3.14]


def main():
    for arm_name, arm_id in (("右臂", RIGHT_ARM_ID), ("左臂", C.LEFT_ARM_ID)):
        arm = LinkerArmA7(robot_id=arm_id, mode="sim")

        def ok(x, y, z, r, p, yw):
            try:
                res = arm.pose_check(x, y, z, roll=r, pitch=p, yaw=yw)
                return bool(res[0] if isinstance(res, (tuple, list)) else res)
            except Exception:
                return False

        print("\n" + "=" * 70)
        print("【%s】姿态空间扫描 (%d 组姿态/点)"
              % (arm_name, len(ROLLS) * len(PITCHES) * len(YAWS)))
        print("=" * 70)

        for name, (x, y, z) in TARGETS.items():
            found = []
            for r, p, yw in itertools.product(ROLLS, PITCHES, YAWS):
                if ok(x, y, z, r, p, yw):
                    found.append((r, p, yw))
            if found:
                print("\n  %s @ (%.3f, %.3f, %.3f)  →  %d 个可行姿态"
                      % (name, x, y, z, len(found)))
                for r, p, yw in found[:5]:
                    # 该姿态下还能抬多高
                    lifts = [h for h in (0.02, 0.05, 0.08, 0.10, 0.15)
                             if ok(x, y, z + h, r, p, yw)]
                    print("      roll=%5.2f pitch=%5.2f yaw=%5.2f   可抬升: %s"
                          % (r, p, yw,
                             ("+%.0fcm" % (max(lifts) * 100)) if lifts else "0(无余量)"))
            else:
                print("\n  %s @ (%.3f, %.3f, %.3f)  →  ✗ 全部姿态无解"
                      % (name, x, y, z))

        try:
            arm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
