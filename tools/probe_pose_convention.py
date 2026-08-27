"""验证 pose_check 的姿态约定 —— 此前所有 out_of_workspace 结论都可能是假的。

实测:home() 成功执行,末端确实到达 pos=(0, 0.2035, -0.6082) rpy=(0,0,0)。
但用 roll=0/pitch=π/yaw=0 调 pose_check,连过该锚点的三条轴都"全不可达"。
=> 机器人能到的点,pose_check 说到不了 => 一直在用错误的姿态参数查询。

注意 home 实测姿态是 rpy=(0,0,0),不是 pitch=π。

本脚本:
  1. 拿 home 锚点自身当探针,试各种姿态,找出 pose_check 认可的姿态
  2. 用认可的姿态重扫工作空间
  3. 顺带确认 pose_check 返回值结构(是否 (ok,msg))

用法:  python -m tools.probe_pose_convention
"""

import itertools
import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)

from agents.nut_picker import config as C
from rabo_robocap import LinkerArmA7


def main():
    arm = LinkerArmA7(robot_id=C.LEFT_ARM_ID, mode="sim")
    arm.home(blocking=True)
    pose = arm.get_pose()
    pos = np.array(pose[0], float)
    rpy = np.array(pose[1], float)
    print("home 实测: pos=(%.4f, %.4f, %.4f)  rpy=(%.4f, %.4f, %.4f)" % (*pos, *rpy))

    def raw(x, y, z, r, p, yw):
        try:
            return arm.pose_check(x, y, z, roll=r, pitch=p, yaw=yw)
        except Exception as e:
            return ("EXC", str(e)[:70])

    print("\n" + "=" * 70)
    print("[1] pose_check 在 home 锚点自身的返回(机器人刚到过这里)")
    print("=" * 70)
    for label, (r, p, yw) in {
        "home 实测 rpy":  tuple(rpy),
        "全零":            (0.0, 0.0, 0.0),
        "pitch=π(此前用的)": (0.0, np.pi, 0.0),
        "pitch=-π":       (0.0, -np.pi, 0.0),
        "pitch=π/2":      (0.0, np.pi / 2, 0.0),
        "roll=π":         (np.pi, 0.0, 0.0),
    }.items():
        print("   %-18s → %r" % (label, raw(*pos, r, p, yw)))

    print("\n" + "=" * 70)
    print("[2] 姿态网格:哪些姿态在 home 锚点被认可")
    print("=" * 70)
    good = []
    vals = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    for r, p, yw in itertools.product(vals, vals, vals):
        res = raw(*pos, r, p, yw)
        ok = bool(res[0]) if isinstance(res, (tuple, list)) and res[0] != "EXC" else False
        if ok:
            good.append((r, p, yw))
    print("   %d/%d 组姿态可达" % (len(good), len(vals) ** 3))
    for g in good[:12]:
        print("      roll=%5.2f pitch=%5.2f yaw=%5.2f" % g)

    if not good:
        print("   ⚠ 无任何姿态被认可 —— pose_check 可能需要别的调用形式")
        print("   尝试位置-only 调用:")
        for form in ("pose_check(x,y,z)", "pose_check([x,y,z])"):
            try:
                r = arm.pose_check(*pos) if "x,y,z)" in form else arm.pose_check(list(pos))
                print("      %s → %r" % (form, r))
            except Exception as e:
                print("      %s → 异常 %s" % (form, str(e)[:70]))
        try:
            arm.shutdown()
        except Exception:
            pass
        return

    # ── 用被认可的姿态重扫 ──────────────────────────────────────────
    r0, p0, y0 = good[0]
    print("\n" + "=" * 70)
    print("[3] 用被认可的姿态 (roll=%.2f pitch=%.2f yaw=%.2f) 重扫工作空间" % (r0, p0, y0))
    print("=" * 70)

    def ok(x, y, z):
        res = raw(x, y, z, r0, p0, y0)
        return bool(res[0]) if isinstance(res, (tuple, list)) and res[0] != "EXC" else False

    ax, ay, az = pos
    for label, gen, idx in (
        ("X", [(round(v, 2), ay, az) for v in np.arange(-0.4, 0.91, 0.05)], 0),
        ("Y", [(ax, round(v, 2), az) for v in np.arange(-0.6, 0.91, 0.05)], 1),
        ("Z", [(ax, ay, round(v, 2)) for v in np.arange(-1.20, 0.21, 0.05)], 2),
    ):
        hit = [g[idx] for g in gen if ok(*g)]
        print("   %s: %s" % (label, "%.2f ~ %.2f (%d 点)" % (min(hit), max(hit), len(hit))
                            if hit else "全不可达"))

    print("\n[4] 各 Z 平面可达点数")
    grid = [(round(x, 2), round(y, 2))
            for x in np.arange(-0.1, 0.71, 0.05)
            for y in np.arange(-0.3, 0.71, 0.05)]
    best = (None, 0)
    for z in np.arange(-1.10, 0.01, 0.05):
        n = sum(ok(x, y, round(z, 2)) for x, y in grid)
        if n:
            print("   z=%6.2f  %3d/%d  %s" % (z, n, len(grid), "#" * int(n / 4)))
        if n > best[1]:
            best = (round(z, 2), n)
    print("   → 最密集 z=%s (%d 点)" % best)

    if best[0] is not None:
        print("\n[5] XY 可达图 @ z=%.2f" % best[0])
        ys = np.arange(-0.30, 0.71, 0.05)
        print("          " + "".join("%6.2f" % y for y in ys))
        for x in np.arange(-0.10, 0.71, 0.05):
            print("    %6.2f%s" % (x, "".join(
                "     %s" % ("#" if ok(round(x, 2), round(y, 2), best[0]) else ".")
                for y in ys)))
        print("\n    (# 可达 / . 不可达; 行=X, 列=Y)")

    try:
        arm.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
