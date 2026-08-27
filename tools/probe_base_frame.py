"""直接问机器人:读末端真实位姿,锚定 base 系原点与朝向。

停止猜坐标系。arm.home() 之后 get_pose() 返回的点必然在 base 系内、
必然可达 —— 一个真实数据点比任何推导都可靠。

对照依据(yizhi-robot/src_remote/workspace_config.py,同款 A7 双臂真机):
    base_frame 原点 = 两肩关节连线中心
    X 正前方(朝向桌子), Y 左正, Z 向上
    table_z = table_height - shoulder_height = 0.74 - 1.252 = -0.512
真机桌面在 base 系是 **负 Z**;仿真当前喂给 IK 的是 +0.284,方向就不对。

本脚本:
  1. home 后读末端位姿(锚点)
  2. 沿 ±X/±Y/±Z 探测可达边界,勾勒工作空间
  3. 用真机 table_point_to_base 思路反推桌面 Z,与视觉结果对照

用法:  python -m tools.probe_base_frame
"""

import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)

from agents.nut_picker import config as C
from rabo_robocap import LinkerArmA7

ARMS = [("左臂", C.LEFT_ARM_ID), ("右臂", "r412d237980e3167577d7aece10f7aedb")]


def dump(obj, label):
    """尽量把 SDK 返回的位姿对象打印清楚。"""
    print("    %s: %r" % (label, obj))
    if hasattr(obj, "__dict__"):
        print("      attrs: %s" % {k: v for k, v in vars(obj).items()
                                   if not k.startswith("_")})


def main():
    for arm_name, arm_id in ARMS:
        print("\n" + "=" * 70)
        print("【%s】 id=%s" % (arm_name, arm_id))
        print("=" * 70)
        try:
            arm = LinkerArmA7(robot_id=arm_id, mode="sim")
        except Exception as e:
            print("  连接失败: %s" % e)
            continue

        # ── 1. 可用方法 ────────────────────────────────────────────
        meths = [m for m in dir(arm) if not m.startswith("_")]
        print("\n[1] SDK 方法: %s" % ", ".join(meths))

        # ── 2. home 后的真实位姿 ───────────────────────────────────
        print("\n[2] home 后读末端位姿(base 系锚点)")
        try:
            arm.home(blocking=True)
            print("    home() 完成")
        except Exception as e:
            print("    home() 失败: %s" % e)

        pose = None
        for name in ("get_pose", "get_tcp_pose", "get_current_pose",
                     "pose", "get_position", "get_state"):
            if hasattr(arm, name):
                try:
                    pose = getattr(arm, name)()
                    dump(pose, name + "()")
                except Exception as e:
                    print("    %s() 异常: %s" % (name, e))

        for name in ("get_joints", "get_joint_positions", "get_joint_states"):
            if hasattr(arm, name):
                try:
                    dump(getattr(arm, name)(), name + "()")
                except Exception as e:
                    print("    %s() 异常: %s" % (name, e))

        # ── 3. 沿各轴探边界 ────────────────────────────────────────
        def ok(x, y, z):
            try:
                r = arm.pose_check(x, y, z, roll=0.0, pitch=np.pi, yaw=0.0)
                return bool(r[0] if isinstance(r, (tuple, list)) else r)
            except Exception:
                return False

        print("\n[3] 工作空间边界扫描(俯视姿态)")
        print("    沿 X 轴 @ y=0, z=-0.3:", end=" ")
        xs = [round(x, 2) for x in np.arange(-0.6, 0.81, 0.05) if ok(round(x, 2), 0.0, -0.30)]
        print("%s" % ("%.2f ~ %.2f" % (min(xs), max(xs)) if xs else "全不可达"))

        print("    沿 Y 轴 @ x=0.35, z=-0.3:", end=" ")
        ys = [round(y, 2) for y in np.arange(-0.6, 0.61, 0.05) if ok(0.35, round(y, 2), -0.30)]
        print("%s" % ("%.2f ~ %.2f" % (min(ys), max(ys)) if ys else "全不可达"))

        print("    沿 Z 轴 @ x=0.35, y=0:", end=" ")
        zs = [round(z, 2) for z in np.arange(-0.9, 0.61, 0.05) if ok(0.35, 0.0, round(z, 2))]
        print("%s" % ("%.2f ~ %.2f" % (min(zs), max(zs)) if zs else "全不可达"))

        # ── 4. 真机约定下的桌面平面 ────────────────────────────────
        print("\n[4] 按真机约定(X 前 / Y 左正 / Z 上,桌面负 Z)扫桌面平面")
        for tz in (-0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25):
            hits = sum(ok(round(x, 2), round(y, 2), tz)
                       for x in np.arange(0.10, 0.61, 0.05)
                       for y in np.arange(-0.30, 0.31, 0.05))
            print("    z=%.2f  可达点 %2d/143" % (tz, hits))

        print("\n[5] XY 扫描 @ 最优 z(真机桌面 -0.512 附近)")
        for tz in (-0.50, -0.45):
            print("\n    ── z = %.2f ──" % tz)
            ys_g = np.arange(-0.35, 0.36, 0.05)
            print("          " + "".join("%6.2f" % y for y in ys_g))
            for x in np.arange(0.10, 0.66, 0.05):
                print("    %6.2f%s" % (x, "".join(
                    "     %s" % ("#" if ok(round(x, 2), round(y, 2), tz) else ".")
                    for y in ys_g)))

        try:
            arm.shutdown()
        except Exception:
            pass

    print("\n(# = 可达, . = 不可达; 行=X 前方, 列=Y 左正)")


if __name__ == "__main__":
    main()
