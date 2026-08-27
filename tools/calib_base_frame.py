"""在正确的 base_link 坐标系下重新标定 head→base。

根因:calibration_runs/gt.json 里的 xyz 与 场景坐标说明.md 的「世界坐标真值」
表逐位相同 —— 标定时把**世界坐标**当成了 base 坐标去拟合 T。

场景真值给出 base 底座在世界系的位置:
    base_origin_world = (-0.6816, 0, 0.752)
因此
    p_base = p_world - base_origin_world

这解释了为什么两条手臂对螺母都 out_of_workspace:喂给 IK 的 X 符号是反的
(-0.305 而非 +0.453)、Z 偏了 0.75m。也解释了为什么箱子"误差 5.9mm" ——
那是在错误坐标系内部自洽,T 把整体平移吸收了,验证脚本测不出来。

本脚本:GT 转到 base 系 → 重解 K/T → 逐点报误差 → pose_check 验证可达性。

用法:  python -m tools.calib_base_frame
"""

import json
import logging

import numpy as np

logging.basicConfig(level=logging.WARNING)

RAW_W, RAW_H = 1920, 1080
BASE_ORIGIN_WORLD = np.array([-0.6816, 0.0, 0.752])   # 场景坐标说明.md


def depth_at(D, u, v, k=2):
    Hd, Wd = D.shape
    u, v = int(round(u)), int(round(v))
    p = D[max(0, v - k):min(Hd, v + k + 1), max(0, u - k):min(Wd, u + k + 1)]
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.median(p)) if p.size else None


def kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def main():
    GT = json.load(open("calibration_runs/gt.json"))
    D = np.load("camera_frames/head_depth_raw.npy")
    Hd, Wd = D.shape

    print("=" * 68)
    print("[1] GT 坐标系换算:world → base")
    print("    base 原点在世界系 = %s" % BASE_ORIGIN_WORLD)
    print("=" * 68)
    pts = []
    for g in GT:
        ur, vr = g["rgb_uv"]
        ud, vd = ur * Wd / RAW_W, vr * Hd / RAW_H
        z = depth_at(D, ud, vd)
        w = np.array(g["xyz"], float)
        b = w - BASE_ORIGIN_WORLD
        pts.append((g["name"], ud, vd, z, b))
        print("%10s  world=(%7.4f,%7.4f,%7.4f)  →  base=(%7.4f,%7.4f,%7.4f)"
              % (g["name"], *w, *b))

    Q = np.array([b for *_, b in pts])
    cx, cy = Wd / 2.0, Hd / 2.0

    def solve(f):
        P = np.array([[(u - cx) * z / f, (v - cy) * z / f, z] for _, u, v, z, _ in pts])
        R, t = kabsch(P, Q)
        e = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
        return float(np.sqrt((e ** 2).mean())), T, e

    # 先看场景标称的 554.3 表现如何,再全域搜一遍
    r554, T554, e554 = solve(554.3)
    best = min((solve(f) + (f,) for f in np.arange(150.0, 1200.0, 0.25)),
               key=lambda x: x[0])
    rmse, T, err, f = best

    print("\n" + "=" * 68)
    print("[2] 标定结果(base 系)")
    print("=" * 68)
    print("  场景标称 f=554.3  → RMSE = %.1f mm" % (r554 * 1000))
    print("  搜索最优 f=%.2f  → RMSE = %.1f mm" % (f, rmse * 1000))
    for (n, *_), e in zip(pts, err):
        print("      %10s  err = %6.1f mm" % (n, e * 1000))

    use_scene = abs(f - 554.3) < 40
    print("\n  → 搜索值%s场景标称值(554.3),%s"
          % ("接近" if use_scene else "偏离",
             "交叉验证通过" if use_scene else "以搜索值为准但需留意"))

    print("\n" + "=" * 68)
    print("[3] pose_check 验证:base 系坐标是否可达")
    print("=" * 68)
    try:
        from agents.nut_picker import config as C
        from rabo_robocap import LinkerArmA7
        for arm_name, arm_id in (("左臂", C.LEFT_ARM_ID),
                                 ("右臂", "r412d237980e3167577d7aece10f7aedb")):
            arm = LinkerArmA7(robot_id=arm_id, mode="sim")
            print("\n  ── %s ──" % arm_name)
            for (n, *_, b) in pts:
                marks = []
                for h in (0.0, 0.05, 0.10):
                    try:
                        r = arm.pose_check(b[0], b[1], b[2] + h, roll=0.0,
                                           pitch=np.pi, yaw=0.0)
                        ok = bool(r[0] if isinstance(r, (tuple, list)) else r)
                    except Exception:
                        ok = False
                    marks.append("+%dcm:%s" % (int(h * 100), "✓" if ok else "✗"))
                print("     %10s base=(%6.3f,%6.3f,%6.3f)  %s"
                      % (n, *b, "  ".join(marks)))
            try:
                arm.shutdown()
            except Exception:
                pass
    except Exception as e:
        print("  (跳过,需仿真运行中: %s)" % e)

    print("\n" + "=" * 68)
    print("# paste into agents/nut_picker/config.py")
    print('HEAD_CAMERA_INTRINSICS = {"fx": %.2f, "fy": %.2f, "cx": %.2f, "cy": %.2f}'
          % (f, f, cx, cy))
    print("HEAD_TO_BASE_T = np.array([")
    for row in T:
        print("    [%s]," % ", ".join("% .8f" % x for x in row))
    print("], dtype=np.float64)")


if __name__ == "__main__":
    main()
