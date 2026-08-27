"""方像素约束下的头部相机重标定。

背景:config.py 里现行的 K 是非方像素(fx=656.84, fy=862.11, 比值 1.31)。
仿真渲染的针孔相机不可能非方像素 —— 那组值是 4 个近似共面的 GT 点
过拟合出来的产物(代码现状.md 自己也记了这个隐患)。

本脚本强制 fx == fy,未知数从 2 个焦距降到 1 个,再用 Kabsch 解刚体变换,
在同样的 4 点 GT 上重解 K 和 T,并与现行 config 对照。

用法:
    python tools/recalibrate.py
    python tools/recalibrate.py --gt calibration_runs/gt.json \
        --depth camera_frames/head_depth_raw.npy
"""

import argparse
import json

import numpy as np

RAW_W, RAW_H = 1920, 1080   # gt.json 中 rgb_uv 所处的坐标系


def depth_at(D, u, v, k=2):
    """取 (2k+1)² 邻域的有效深度中位数,抗单像素噪声。"""
    Hd, Wd = D.shape
    u, v = int(round(u)), int(round(v))
    p = D[max(0, v - k):min(Hd, v + k + 1), max(0, u - k):min(Wd, u + k + 1)]
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.median(p)) if p.size else None


def kabsch(P, Q):
    """求 R, t 使 R @ P + t ≈ Q(最小二乘刚体对齐)。"""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def solve(pts, Q, f, cx, cy):
    """给定焦距与主点,反投影到相机系并对齐到 base 系。返回 (rmse, T, 逐点误差)。"""
    P = np.array([[(u - cx) * z / f, (v - cy) * z / f, z] for _, u, v, z, _ in pts])
    R, t = kabsch(P, Q)
    e = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return float(np.sqrt((e ** 2).mean())), T, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="calibration_runs/gt.json")
    ap.add_argument("--depth", default="camera_frames/head_depth_raw.npy")
    ap.add_argument("--fmin", type=float, default=150.0)
    ap.add_argument("--fmax", type=float, default=1200.0)
    ap.add_argument("--step", type=float, default=0.25)
    args = ap.parse_args()

    GT = json.load(open(args.gt))
    D = np.load(args.depth)
    Hd, Wd = D.shape
    print("depth: shape=%s  有效=%.1f%%  range=%.4f~%.4f"
          % (D.shape, 100 * (np.isfinite(D) & (D > 0)).mean(),
             np.nanmin(D[D > 0]), np.nanmax(D)))

    pts = []
    for g in GT:
        ur, vr = g["rgb_uv"]
        ud, vd = ur * Wd / RAW_W, vr * Hd / RAW_H
        z = depth_at(D, ud, vd)
        pts.append((g["name"], ud, vd, z, np.array(g["xyz"], float)))
        print("%10s  depth_px=(%6.1f,%6.1f)  z=%.4f  gt=%s"
              % (g["name"], ud, vd, z, g["xyz"]))

    Q = np.array([g for *_, g in pts])
    cx, cy = Wd / 2.0, Hd / 2.0

    best = None
    for f in np.arange(args.fmin, args.fmax, args.step):
        r, T, e = solve(pts, Q, f, cx, cy)
        if best is None or r < best[0]:
            best = (r, f, T, e)
    rmse, f, T, err = best

    print("\n" + "=" * 62)
    print("[方像素解] f = %.2f   RMSE = %.1f mm" % (f, rmse * 1000))
    for (n, *_), e in zip(pts, err):
        print("   %10s  err = %6.1f mm" % (n, e * 1000))

    # 与现行 config 对照
    try:
        import agents.nut_picker.config as C
        K = C.HEAD_CAMERA_INTRINSICS
        Pc = np.array([[(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z]
                       for _, u, v, z, _ in pts])
        ec = np.linalg.norm(
            (Pc @ C.HEAD_TO_BASE_T[:3, :3].T + C.HEAD_TO_BASE_T[:3, 3]) - Q, axis=1)
        print("\n[当前 config] fx=%s fy=%s  mean=%.1f mm  max=%.1f mm"
              % (K["fx"], K["fy"], ec.mean() * 1000, ec.max() * 1000))
    except Exception as e:
        print("\n[当前 config] 对照跳过: %s" % e)

    print("\n" + "=" * 62)
    print("# paste into agents/nut_picker/config.py")
    print('HEAD_CAMERA_INTRINSICS = {"fx": %.2f, "fy": %.2f, "cx": %.2f, "cy": %.2f}'
          % (f, f, cx, cy))
    print("HEAD_TO_BASE_T = np.array([")
    for row in T:
        print("    [%s]," % ", ".join("% .8f" % x for x in row))
    print("], dtype=np.float64)")


if __name__ == "__main__":
    main()
