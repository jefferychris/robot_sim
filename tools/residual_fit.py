"""残差修正:在现有 K/T 之上叠加一个仿射修正,把系统性偏差吃掉。

诊断依据:方像素重标定后误差仍有 53.5mm,且与物体大小强相关
(box 8.9mm < nutB 24.3 < nutC 59.4 < nutA 85.3)。随机标定误差不会
呈现这种规律 —— 说明 GT 的 rgb_uv(标在物体中心)与深度采样点(物体顶面)
存在系统性错配,物体越大偏差越大。

继续拟合 K/T 只会拟合到错的数据上。改用残差修正:
    p_corrected = A @ p_predicted + b
A(3x3) + b(3) 共 12 个自由度,4 个点提供 12 个方程 —— 恰定。
恰定意味着训练误差必然为 0,所以脚本同时做 leave-one-out 交叉验证,
用留出点的误差评估真实泛化能力。若 LOO 误差不显著优于 53mm,说明
这条路也不成立,应转向腕部相机方案。

用法:  python tools/residual_fit.py
"""

import json

import numpy as np

from agents.nut_picker import config as C
from agents.nut_picker import geometry

RAW_W, RAW_H = 1920, 1080


def predict(D, u_raw, v_raw):
    """用现行 config 的 K/T 把 GT 像素反投影到 base 系。"""
    Hd, Wd = D.shape
    ud, vd = u_raw * Wd / RAW_W, v_raw * Hd / RAW_H
    k = 2
    p = D[max(0, int(vd) - k):int(vd) + k + 1, max(0, int(ud) - k):int(ud) + k + 1]
    p = p[np.isfinite(p) & (p > 0)]
    if not p.size:
        return None
    z = float(np.median(p))
    K, T = C.HEAD_CAMERA_INTRINSICS, C.HEAD_TO_BASE_T
    pc = np.array([(ud - K["cx"]) * z / K["fx"], (vd - K["cy"]) * z / K["fy"], z])
    return T[:3, :3] @ pc + T[:3, 3]


def fit(P, Q):
    """最小二乘解 A, b 使 A @ p + b ≈ q。"""
    X = np.hstack([P, np.ones((len(P), 1))])       # (n,4)
    sol, *_ = np.linalg.lstsq(X, Q, rcond=None)     # (4,3)
    return sol[:3].T, sol[3]


def main():
    GT = json.load(open("calibration_runs/gt.json"))
    D = np.load("camera_frames/head_depth_raw.npy")

    names, P, Q = [], [], []
    for g in GT:
        p = predict(D, *g["rgb_uv"])
        if p is None:
            print("跳过(无深度): %s" % g["name"])
            continue
        names.append(g["name"])
        P.append(p)
        Q.append(g["xyz"])
    P, Q = np.array(P), np.array(Q, float)

    print("=" * 64)
    print("[修正前] 现行 config 的逐点误差")
    print("=" * 64)
    e0 = np.linalg.norm(P - Q, axis=1)
    for n, p, q, e in zip(names, P, Q, e0):
        print("%10s  pred=(%.3f,%.3f,%.3f)  gt=(%.3f,%.3f,%.3f)  err=%6.1f mm"
              % (n, *p, *q, e * 1000))
    print("%10s  mean=%.1f mm  max=%.1f mm" % ("", e0.mean() * 1000, e0.max() * 1000))

    A, b = fit(P, Q)
    e1 = np.linalg.norm((P @ A.T + b) - Q, axis=1)
    print("\n[修正后-训练集] mean=%.1f mm (恰定系统,必然≈0,不代表泛化)"
          % (e1.mean() * 1000))

    # leave-one-out:这才是真实泛化能力
    print("\n" + "=" * 64)
    print("[LOO 交叉验证] 留一点不参与拟合,看它的预测误差")
    print("=" * 64)
    loo = []
    for i in range(len(P)):
        m = np.ones(len(P), bool)
        m[i] = False
        try:
            Ai, bi = fit(P[m], Q[m])
            e = float(np.linalg.norm((Ai @ P[i] + bi) - Q[i]))
        except np.linalg.LinAlgError:
            e = float("nan")
        loo.append(e)
        print("  留出 %10s  err = %7.1f mm  (原 %5.1f mm)"
              % (names[i], e * 1000, e0[i] * 1000))
    loo = np.array(loo)
    print("\n  LOO 平均 = %.1f mm   vs   修正前 %.1f mm" % (np.nanmean(loo) * 1000,
                                                           e0.mean() * 1000))
    if np.nanmean(loo) < e0.mean() * 0.6:
        print("  → 修正有效,建议采用")
    else:
        print("  → 修正无泛化能力(4 点拟合 12 自由度过拟合),不要采用")
        print("  → 建议改用腕部相机(eye-in-hand),或增加 GT 点")

    print("\n" + "=" * 64)
    print("# 若采用,贴进 agents/nut_picker/config.py:")
    print("RESIDUAL_A = np.array([")
    for r in A:
        print("    [%s]," % ", ".join("% .8f" % x for x in r))
    print("], dtype=np.float64)")
    print("RESIDUAL_B = np.array([%s], dtype=np.float64)"
          % ", ".join("% .8f" % x for x in b))


if __name__ == "__main__":
    main()
