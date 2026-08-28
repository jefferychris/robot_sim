"""工作空间探测：用 arm.pose_check() 批量预检可达性。

背景：move_to 对桌面高度 (z≈-0.47) 报 out_of_workspace，但 move_joints 能到。
pose_check(x,y,z,roll,pitch,yaw) -> (bool, str) 可以在不动机械臂的前提下预检，
用它扫出真实的工作空间边界，替代之前反复试 move_to 的做法。

用法（平台终端）：
    cd /workspace/agent_system && source /opt/rabo-venvs/agent_system/bin/activate
    python3 tools/probe_workspace.py
"""

import math
import sys

from rabo_robocap import LinkerArmA7

ARM_ID = "r412d237980e3167577d7aece10f7aedb"

# 三颗螺母 + 三个格子的实测目标位置（SDK 坐标）
TARGETS = [
    ("nut_big",    0.388, 0.060, -0.472),
    ("nut_medium", 0.338, 0.169, -0.467),
    ("nut_small",  0.447, 0.102, -0.472),
]

# 俯视抓取姿态
RPY = (0.0, math.pi, 0.0)


def check(arm, label, x, y, z, rpy=RPY):
    roll, pitch, yaw = rpy
    try:
        ok, msg = arm.pose_check(x, y, z, roll, pitch, yaw)
    except Exception as e:
        return False, f"EXC {e}"
    return ok, msg


def main():
    print("初始化右臂 ...")
    arm = LinkerArmA7(robot_id=ARM_ID, mode="sim")
    print("当前关节角:", arm.get_joint_angles())
    print("当前末端位姿:", arm.get_pose())
    print()

    # ── 1. 直接检目标点 ────────────────────────────────────────────
    print("=" * 64)
    print("1) 目标点可达性（俯视姿态 pitch=pi）")
    print("=" * 64)
    for label, x, y, z in TARGETS:
        ok, msg = check(arm, label, x, y, z)
        print(f"  {label:12s} ({x:.3f}, {y:.3f}, {z:.3f})  ->  {ok}  {msg}")
    print()

    # ── 2. 目标点上方不同高度 ──────────────────────────────────────
    print("=" * 64)
    print("2) 沿 Z 扫描（找可达的最低高度 = 能否够到桌面）")
    print("=" * 64)
    for label, x, y, _z in TARGETS:
        line = [f"  {label:12s}"]
        for z in [-0.50, -0.45, -0.40, -0.30, -0.20, -0.10, 0.0, 0.1, 0.2, 0.3]:
            ok, _ = check(arm, label, x, y, z)
            line.append(f"{z:+.2f}:{'Y' if ok else '.'}")
        print(" ".join(line))
    print()

    # ── 3. 换姿态再试目标点 ────────────────────────────────────────
    print("=" * 64)
    print("3) 不同抓取姿态下的目标点可达性")
    print("=" * 64)
    poses = [
        ("俯视 pitch=pi",      (0.0, math.pi, 0.0)),
        ("俯视 pitch=-pi/2",   (0.0, -math.pi / 2, 0.0)),
        ("俯视 pitch=pi/2",    (0.0, math.pi / 2, 0.0)),
        ("水平 pitch=0",       (0.0, 0.0, 0.0)),
        ("俯视+yaw90",         (0.0, math.pi, math.pi / 2)),
        ("斜 45°",             (0.0, math.pi * 0.75, 0.0)),
    ]
    for pname, rpy in poses:
        res = []
        for label, x, y, z in TARGETS:
            ok, _ = check(arm, label, x, y, z, rpy)
            res.append(f"{label.split('_')[1][:3]}:{'Y' if ok else '.'}")
        print(f"  {pname:20s} " + "  ".join(res))
    print()

    # ── 4. XY 平面粗扫（固定桌面高度）─────────────────────────────
    print("=" * 64)
    print("4) XY 平面可达域（z=-0.47 桌面高度，俯视姿态）")
    print("=" * 64)
    zs = -0.47
    ys = [round(-0.1 + 0.05 * i, 2) for i in range(9)]   # -0.10 .. 0.30
    xs = [round(0.20 + 0.05 * i, 2) for i in range(9)]   # 0.20 .. 0.60
    print("       " + " ".join(f"{y:+.2f}" for y in ys))
    for x in xs:
        row = [f"x={x:.2f}"]
        for y in ys:
            ok, _ = check(arm, "scan", x, y, zs)
            row.append("  Y  " if ok else "  .  ")
        print(" ".join(row))
    print()
    print("(Y=可达  .=不可达)")

    arm.shutdown()


if __name__ == "__main__":
    main()
