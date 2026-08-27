"""以 home 位姿为锚点,在其邻域扫描工作空间并定位桌面。

上一轮实测锚点(左臂 home):
    get_pose() = ((0.0, 0.2035, -0.6082), (0.0, 0.0, 0.0))
=> base 原点在立柱/肩部高处;Y 左正(左臂在 +0.2);桌面是**负 Z 且低于 -0.61**。
上一轮扫描全 0 是因为参数选错(固定 y=0 偏离左臂中心 +0.2;Z 只扫到 -0.55)。

视觉当前输出 Z=+0.284 —— 符号与量级都不对,这就是全线 out_of_workspace 的根因。

本脚本以锚点为中心重扫,并输出场景世界坐标 → base 系的换算候选。

用法:  python -m tools.probe_around_home
"""

import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)

from agents.nut_picker import config as C
from rabo_robocap import LinkerArmA7

ARMS = [("左臂", C.LEFT_ARM_ID), ("右臂", "r412d237980e3167577d7aece10f7aedb")]

# 场景坐标说明.md 的世界坐标真值
WORLD = {
    "nut_big":    np.array([-0.2286, -0.0999, 0.2815]),
    "nut_medium": np.array([-0.3413, -0.1710, 0.2806]),
    "nut_small":  np.array([-0.2975, -0.0527, 0.2868]),
    "box":        np.array([-0.3104,  0.2586, 0.2996]),
}


def main():
    for arm_name, arm_id in ARMS:
        print("\n" + "=" * 72)
        print("【%s】" % arm_name)
        print("=" * 72)
        arm = LinkerArmA7(robot_id=arm_id, mode="sim")
        try:
            arm.home(blocking=True)
            p = arm.get_pose()
            pos = np.array(p[0], float)
            print("  home 锚点: pos=(%.4f, %.4f, %.4f)  rpy=%s" % (*pos, p[1]))
        except Exception as e:
            print("  取锚点失败: %s" % e)
            pos = np.array([0.0, 0.2035, -0.6082])

        def ok(x, y, z, pitch=np.pi):
            try:
                r = arm.pose_check(x, y, z, roll=0.0, pitch=pitch, yaw=0.0)
                return bool(r[0] if isinstance(r, (tuple, list)) else r)
            except Exception:
                return False

        ax, ay, az = pos

        # ── 以锚点为中心,逐轴扫 ────────────────────────────────────
        print("\n[1] 过锚点的三轴可达区间(俯视姿态)")
        for label, gen in (
            ("X", [(round(v, 2), ay, az) for v in np.arange(-0.3, 0.81, 0.05)]),
            ("Y", [(ax, round(v, 2), az) for v in np.arange(-0.5, 0.81, 0.05)]),
            ("Z", [(ax, ay, round(v, 2)) for v in np.arange(-1.10, 0.01, 0.05)]),
        ):
            hit = [g for g in gen if ok(*g)]
            if hit:
                idx = {"X": 0, "Y": 1, "Z": 2}[label]
                vals = [h[idx] for h in hit]
                print("    %s: %.2f ~ %.2f  (共 %d 点)" % (label, min(vals), max(vals), len(vals)))
            else:
                print("    %s: 全不可达" % label)

        # ── 找桌面:哪个 Z 平面可达点最多 ───────────────────────────
        print("\n[2] 各 Z 平面可达点数(x 0.0~0.6, y -0.2~0.6)")
        grid = [(round(x, 2), round(y, 2))
                for x in np.arange(0.0, 0.61, 0.05)
                for y in np.arange(-0.2, 0.61, 0.05)]
        best_z, best_n = None, 0
        for z in np.arange(-1.05, -0.24, 0.05):
            n = sum(ok(x, y, round(z, 2)) for x, y in grid)
            bar = "#" * int(n / 4)
            print("    z=%6.2f  %3d/%d  %s" % (z, n, len(grid), bar))
            if n > best_n:
                best_z, best_n = round(z, 2), n
        print("    → 可达点最密集: z = %.2f (%d 点)" % (best_z, best_n))

        # ── 该平面的 XY 图 ─────────────────────────────────────────
        if best_z is not None:
            print("\n[3] XY 可达图 @ z=%.2f" % best_z)
            ys = np.arange(-0.25, 0.66, 0.05)
            print("          " + "".join("%6.2f" % y for y in ys))
            for x in np.arange(0.0, 0.66, 0.05):
                print("    %6.2f%s" % (x, "".join(
                    "     %s" % ("#" if ok(round(x, 2), round(y, 2), best_z) else ".")
                    for y in ys)))
            print("\n    (# 可达 / . 不可达; 行=X, 列=Y)")

        # ── 世界→base 换算候选 ─────────────────────────────────────
        print("\n[4] 世界坐标 → base 换算候选")
        print("    场景世界坐标里桌面物体 z≈0.28;若 base 系桌面在 z=%.2f," % (best_z or -0.7))
        print("    则 z 方向偏移 ≈ %.3f" % ((best_z or -0.7) - 0.28))
        for name, w in WORLD.items():
            print("      %10s world=(%7.4f,%7.4f,%7.4f)" % (name, *w))

        try:
            arm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
