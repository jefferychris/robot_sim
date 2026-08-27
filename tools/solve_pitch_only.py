"""兜底:约束"相机只俯视"重解外参,把未知数从 6 个降到 3 个。

场景 JSON 的相机挂载 rpy=[0, 0.8, 0] —— 只有俯仰,没有翻滚和偏航
(截图里相机装在立柱顶端朝下看,与 0.8rad≈45.8° 吻合)。

现有 4 点 Kabsch 解 6 个旋转自由度 + 3 个平移,而 4 点几乎共面,必然过拟合。
约束成"只有俯仰 + 平移"后,未知数 = pitch + tx,ty,tz = 4 个,4 点足够定解。

同时用 base 系 GT(世界坐标 - base 原点)而非世界坐标本身。
base 原点由 home 锚点实测 (0, 0.2035, -0.6082) 提供约束。

用法:  python -m tools.solve_pitch_only
"""

import json
import math

import numpy as np

RAW_W, RAW_H = 1920, 1080


def depth_at(D, u, v, k=2):
    Hd, Wd = D.shape
    u, v = int(round(u)), int(round(v))
    p = D[max(0, v-k):min(Hd, v+k+1), max(0, u-k):min(Wd, u+k+1)]
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.median(p)) if p.size else None


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    GT = json.load(open("calibration_runs/gt.json"))
    D = np.load("camera_frames/head_depth_raw.npy")
    Hd, Wd = D.shape

    pts = []
    for g in GT:
        ur, vr = g["rgb_uv"]
        ud, vd = ur * Wd / RAW_W, vr * Hd / RAW_H
        z = depth_at(D, ud, vd)
        pts.append((g["name"], ud, vd, z, np.array(g["xyz"], float)))

    Q_world = np.array([p[4] for p in pts])
    cx, cy = Wd / 2.0, Hd / 2.0

    print("=" * 72)
    print("约束搜索:相机只俯视(pitch),平移自由;GT 用世界坐标,同时解 base 偏移")
    print("=" * 72)

    best = None
    for f in np.arange(400.0, 900.0, 2.0):
        P = np.array([[(u-cx)*z/f, (v-cy)*z/f, z] for _, u, v, z, _ in pts])
        for pitch in np.arange(0.3, 1.4, 0.01):
            for yaw in (0.0, math.pi/2, math.pi, -math.pi/2):
                # 相机光轴朝下:先绕 X 转 -90° 把光轴从 +Z 转到 -Z 方向,再俯仰
                R = Rz(yaw) @ Ry(pitch) @ Rx(-math.pi/2)
                Pr = P @ R.T
                t = (Q_world - Pr).mean(0)          # 平移 = 残差均值
                e = np.linalg.norm(Pr + t - Q_world, axis=1)
                rmse = float(np.sqrt((e**2).mean()))
                if best is None or rmse < best[0]:
                    best = (rmse, f, pitch, yaw, R, t, e)

    rmse, f, pitch, yaw, R, t, err = best
    print("\n  f     = %.1f" % f)
    print("  pitch = %.3f rad (%.1f°)   [场景 JSON 标称 0.8 rad = 45.8°]" % (pitch, math.degrees(pitch)))
    print("  yaw   = %.3f rad" % yaw)
    print("  t     = (%.4f, %.4f, %.4f)" % tuple(t))
    print("  RMSE  = %.1f mm     [4点Kabsch: 43.7mm]" % (rmse*1000))
    for (n, *_), e in zip(pts, err):
        print("      %10s  %6.1f mm" % (n, e*1000))

    close = abs(pitch - 0.8) < 0.15
    print("\n  → 解出的 pitch %s场景标称 0.8rad %s"
          % ("接近" if close else "偏离", "✓ 交叉验证通过" if close else "⚠ 需留意"))

    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t
    print("\n" + "=" * 72)
    print("# 这是 相机→世界。base 系还需减去 base 原点(home 锚点可反推)")
    print("=" * 72)
    print("HEAD_TO_WORLD_T = np.array([")
    for row in T:
        print("    [%s]," % ", ".join("% .8f" % v for v in row))
    print("], dtype=np.float64)")


if __name__ == "__main__":
    main()
