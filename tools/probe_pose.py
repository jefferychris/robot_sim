"""姿态搜索：固定螺母 XYZ，扫 roll/pitch/yaw 找能解出 IK 的姿态。

背景：probe_workspace 显示三颗螺母在 z=-0.47 时俯视姿态全部 out_of_workspace，
但把 z 抬到 -0.10 就可达，且 pitch=0 时 nut_medium 可达 —— 说明限制来自
IK 解算 + 姿态约束，不是几何够不到（末端 home 位就在 z=-0.61）。

本脚本网格搜索姿态空间，找出每颗螺母在真实桌面高度下可行的抓取姿态。

用法（平台终端）：
    cd /workspace/agent_system && source /opt/rabo-venvs/agent_system/bin/activate
    python3 tools/probe_pose.py
"""

import math
from rabo_robocap import LinkerArmA7

ARM_ID = "r412d237980e3167577d7aece10f7aedb"

TARGETS = [
    ("nut_big",    0.388, 0.060, -0.472),
    ("nut_medium", 0.338, 0.169, -0.467),
    ("nut_small",  0.447, 0.102, -0.472),
]

PI = math.pi


def main():
    arm = LinkerArmA7(robot_id=ARM_ID, mode="sim")
    print("末端 home 位姿:", arm.get_pose())
    print()

    def ck(x, y, z, r, p, yw):
        try:
            ok, msg = arm.pose_check(x, y, z, r, p, yw)
            return ok, msg
        except Exception as e:
            return False, str(e)

    # ── 1. pitch 细扫（roll=0, yaw=0）──────────────────────────────
    print("=" * 70)
    print("1) pitch 细扫  (roll=0, yaw=0, 桌面高度)")
    print("=" * 70)
    pitches = [round(-PI + i * PI / 12, 3) for i in range(25)]  # -pi..pi 步长15°
    for label, x, y, z in TARGETS:
        hits = []
        for p in pitches:
            ok, _ = ck(x, y, z, 0.0, p, 0.0)
            if ok:
                hits.append(round(math.degrees(p)))
        print(f"  {label:12s} 可行 pitch(deg): {hits if hits else '无'}")
    print()

    # ── 2. yaw 细扫（pitch 取上面找到的，或 0）────────────────────
    print("=" * 70)
    print("2) yaw 细扫  (roll=0, pitch=0, 桌面高度)")
    print("=" * 70)
    yaws = [round(-PI + i * PI / 12, 3) for i in range(25)]
    for label, x, y, z in TARGETS:
        hits = []
        for yw in yaws:
            ok, _ = ck(x, y, z, 0.0, 0.0, yw)
            if ok:
                hits.append(round(math.degrees(yw)))
        print(f"  {label:12s} 可行 yaw(deg): {hits if hits else '无'}")
    print()

    # ── 3. roll 细扫 ──────────────────────────────────────────────
    print("=" * 70)
    print("3) roll 细扫  (pitch=0, yaw=0, 桌面高度)")
    print("=" * 70)
    rolls = [round(-PI + i * PI / 12, 3) for i in range(25)]
    for label, x, y, z in TARGETS:
        hits = []
        for r in rolls:
            ok, _ = ck(x, y, z, r, 0.0, 0.0)
            if ok:
                hits.append(round(math.degrees(r)))
        print(f"  {label:12s} 可行 roll(deg): {hits if hits else '无'}")
    print()

    # ── 4. 粗网格 roll×pitch×yaw 全搜（30° 步长）─────────────────
    print("=" * 70)
    print("4) 全姿态粗搜 (30° 步长, 桌面高度) —— 每颗螺母列前 12 个可行解")
    print("=" * 70)
    step = PI / 6
    grid = [round(-PI + i * step, 3) for i in range(13)]
    for label, x, y, z in TARGETS:
        sols = []
        for r in grid:
            for p in grid:
                for yw in grid:
                    ok, _ = ck(x, y, z, r, p, yw)
                    if ok:
                        sols.append((round(math.degrees(r)),
                                     round(math.degrees(p)),
                                     round(math.degrees(yw))))
        print(f"  {label:12s} 可行解 {len(sols)} 个")
        for s in sols[:12]:
            print(f"      roll={s[0]:+4d}  pitch={s[1]:+4d}  yaw={s[2]:+4d}")
        if not sols:
            print("      （无解 —— 该高度下 move_to 不可用，需走 move_joints）")
    print()

    # ── 5. 逐步降低 Z，找每颗螺母的最低可达高度 ──────────────────
    print("=" * 70)
    print("5) 最低可达高度（任意姿态，30° 粗搜）")
    print("=" * 70)
    for label, x, y, _z in TARGETS:
        best = None
        for zi in range(0, 26):
            z = -0.10 - zi * 0.02      # -0.10 往下到 -0.60
            found = None
            for r in (0.0, PI):
                for p in grid:
                    for yw in grid:
                        ok, _ = ck(x, y, z, r, p, yw)
                        if ok:
                            found = (round(math.degrees(r)),
                                     round(math.degrees(p)),
                                     round(math.degrees(yw)))
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                best = (z, found)
            else:
                break
        if best:
            z, (r, p, yw) = best
            print(f"  {label:12s} 最低 z={z:+.2f}  姿态 roll={r} pitch={p} yaw={yw}")
        else:
            print(f"  {label:12s} 无可达高度")

    arm.shutdown()


if __name__ == "__main__":
    main()
