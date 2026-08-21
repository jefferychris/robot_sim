"""相机标定:从已知 ground truth + 检测像素 + 原始深度反推内参 + head→base。

────────────────────────────────────────────────────────────────────────
什么时候用:
  当前 config.py 里的 HEAD_CAMERA_INTRINSICS 是占位 {fx:615, ...},
  HEAD_TO_BASE_T 是用 4 个 GT 点凑出来的「专用」4x4,螺母换位置就不准。
  用本脚本重新标定,得到真正的内参 + 通用 head→base 4x4,后续都准。

工作流(2 步):

  步骤 1: 在平台侧跑一次 camera_demo,带环境变量保存原始深度:
      SAVE_DEPTH_NPY=1 python main.py camera_demo
    输出文件:
      camera_frames/head_rgb.png         (RGB, 1920x1080)
      camera_frames/head_depth.png       (灰度 PNG, 已归一化)
      camera_frames/head_depth_raw.npy   ★ float32 米数

  步骤 2: 跑本脚本,提供 ground truth:
      python -m agents.nut_picker.calibrate \
          --gt-json gt.json

  gt.json 格式(每个对象含 name + ground_truth XYZ + 像素位置):
    [
      {"name": "nutA", "xyz": [-0.2286, -0.0999, 0.2815], "rgb_uv": [1087, 700]},
      {"name": "nutB", "xyz": [-0.3413, -0.1710, 0.2806], "rgb_uv": [1340, 793]},
      ...
    ]

  输出:打印新的 HEAD_CAMERA_INTRINSICS 和 HEAD_TO_BASE_T,用户贴回 config.py。

────────────────────────────────────────────────────────────────────────
数学:
  对于每个 GT 点 i (世界 XYZ) 和检测像素 (u, v) + 深度 d:
    1. RGB 像素 → depth 像素(按比例缩放)
    2. 针孔反投影 → camera 坐标 (x_cam, y_cam, d)
    3. PnP 求解:求解 T (4×4 head→base) 使得 T @ [x_cam, y_cam, d, 1] = [X, Y, Z]

  联合求解:
    - 内参 (fx, fy, cx, cy):用 PnP 内参迭代求解
    - 至少需要 4 个非共面点;5+ 更好
────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

# 让本脚本能以模块方式运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.camera_demo.config import CAMERAS, SAVE_DIR

logger = logging.getLogger("nut_picker.calibrate")


# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════
def rgb_uv_to_depth_uv(u_rgb: float, v_rgb: float,
                       rgb_shape: tuple, depth_shape: tuple) -> tuple:
    """RGB 像素 → depth 像素坐标(按比例缩放)。"""
    H, W = rgb_shape
    H_d, W_d = depth_shape
    return (u_rgb * W_d / W, v_rgb * H_d / H)


def get_depth_at_raw(depth_raw: np.ndarray, u_d: float, v_d: float,
                     radius: int = 1) -> float:
    """从 raw float32 深度图取邻域有效深度中位数(米)。

    radius=0 → 单像素;radius=1 → 3x3 邻域(默认,稳健降噪);
    radius=2 → 5x5 邻域。邻域内 inf/nan/<=0 视为无效,不参与中位数。
    """
    h, w = depth_raw.shape[:2]
    ui, vi = int(round(u_d)), int(round(v_d))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    if radius == 0:
        val = float(depth_raw[vi, ui])
        return val if np.isfinite(val) and val > 0 else None
    vals = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            uu, vv = ui + du, vi + dv
            if 0 <= uu < w and 0 <= vv < h:
                v = float(depth_raw[vv, uu])
                if np.isfinite(v) and v > 0:
                    vals.append(v)
    if not vals:
        return None
    return float(np.median(vals))


# ══════════════════════════════════════════════════════════════════════
#  PnP 求解
# ══════════════════════════════════════════════════════════════════════
def solve_pnp(points: list, ransac_iters: int = 50,
              outlier_thresh_m: float = 0.10,
              search_cx_cy: bool = False) -> tuple:
    """网格搜索 fx, fy + SVD 解 R,t,RANSAC 排除 outlier GT。

    主点 (cx, cy) 默认固定 = 图像中心(物理上绝大多数相机成立)。
    设 search_cx_cy=True 会加入 cx, cy 网格搜索,但 4 点 GT 下会过拟合,谨慎用。

    策略:
      step 1: 网格搜 fx, fy
      step 2: (可选)固定 fx, fy 搜 cx, cy
      step 3: 用最优 K 全点 SVD → 排除误差 > outlier_thresh_m 的点 → 重做 SVD

    输入 points: list of {"uv_depth": (u_d, v_d), "depth_m": d, "xyz_gt": (X, Y, Z), "depth_shape"}
    返回: (K, T, mean_err, inlier_mask)
    """
    if len(points) < 4:
        raise ValueError(f"至少需要 4 个点,当前 {len(points)}")

    H_d, W_d = points[0]["depth_shape"]
    cx0 = W_d / 2.0
    cy0 = H_d / 2.0

    def estimate_cam(fx, fy, cx, cy, pts):
        cam = np.zeros((len(pts), 3))
        for i, p in enumerate(pts):
            u_d, v_d = p["uv_depth"]
            d = p["depth_m"]
            cam[i] = [(u_d - cx) * d / fx, (v_d - cy) * d / fy, d]
        return cam

    def rigid_transform(src, dst):
        """SVD 解 src → dst 刚体变换 R, t 使 R @ src + t = dst。"""
        src_c = src - src.mean(axis=0)
        dst_c = dst - dst.mean(axis=0)
        H = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = dst.mean(axis=0) - R @ src.mean(axis=0)
        return R, t

    def err_for(fx, fy, cx, cy, pts):
        cam = estimate_cam(fx, fy, cx, cy, pts)
        world = np.array([p["xyz_gt"] for p in pts])
        R, t = rigid_transform(cam, world)
        pred = (R @ cam.T).T + t
        err = np.linalg.norm(pred - world, axis=1)
        return err.mean(), R, t, err

    # grid 范围
    fxs = np.linspace(W_d / 2, W_d * 2.5, 20)
    fys = np.linspace(H_d / 2, H_d * 2.5, 20)

    best = (1e9, None, None, cx0, cy0, None, None, None)  # (err, fx, fy, cx, cy, R, t, per_pt_err)

    # step 1: 搜 fx, fy(cx, cy = 中心)
    for fx in fxs:
        for fy in fys:
            e, R, t, per_e = err_for(fx, fy, cx0, cy0, points)
            if e < best[0]:
                best = (e, fx, fy, cx0, cy0, R, t, per_e)

    err, fx, fy, cx, cy, R, t, per_e = best

    # step 2 (可选): 固定 fx, fy 搜 cx, cy(谨慎:4 点 GT 易过拟合)
    if search_cx_cy:
        cx_range = np.linspace(cx0 - W_d * 0.15, cx0 + W_d * 0.15, 11)
        cy_range = np.linspace(cy0 - H_d * 0.15, cy0 + H_d * 0.15, 11)
        for cx_ in cx_range:
            for cy_ in cy_range:
                e, R_, t_, per_e_ = err_for(fx, fy, cx_, cy_, points)
                if e < best[0]:
                    best = (e, fx, fy, cx_, cy_, R_, t_, per_e_)
                    cx, cy = cx_, cy_

    err, fx, fy, cx, cy, R, t, per_e = best

    # step 3: RANSAC — 排除误差 > outlier_thresh_m 的点(GT 标错/箱 yaw 等)后重做 SVD
    inlier_mask = per_e < outlier_thresh_m
    n_inliers = int(inlier_mask.sum())

    # 准备返回值
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    if n_inliers >= 4 and n_inliers < len(points):
        # 有 outlier,排除后重做(用全点同一组 fx/fy/cx/cy,只换点集)
        inlier_pts = [points[i] for i in range(len(points)) if inlier_mask[i]]
        e2, R2, t2, per_e2 = err_for(fx, fy, cx, cy, inlier_pts)
        if e2 < err:
            err, R, t = e2, R2, t2
            T[:3, :3] = R
            T[:3, 3] = t
            full_mask = np.zeros(len(points), dtype=bool)
            for i, p in enumerate(points):
                if p in inlier_pts:
                    full_mask[i] = True
            inlier_mask = full_mask
        else:
            inlier_mask = np.ones(len(points), dtype=bool)
    else:
        inlier_mask = np.ones(len(points), dtype=bool)

    return K, T, float(err), inlier_mask


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Nut picker camera calibration")
    parser.add_argument(
        "--gt-json",
        required=True,
        help="JSON 列表,每项 {name, xyz, rgb_uv}",
    )
    parser.add_argument(
        "--rgb-path",
        default=os.path.join(SAVE_DIR, "head_rgb.png"),
        help="RGB 图路径",
    )
    parser.add_argument(
        "--depth-raw-path",
        default=os.path.join(SAVE_DIR, "head_depth_raw.npy"),
        help="原始深度 .npy 路径(float32 米数, 优先用)",
    )
    parser.add_argument(
        "--depth-png-path",
        default=os.path.join(SAVE_DIR, "head_depth.png"),
        help="归一化深度 PNG 路径(无 raw 时的 fallback;精度低)",
    )
    parser.add_argument(
        "--depth-near-m",
        type=float,
        default=None,
        help="深度 PNG 最亮像素(255)对应的米数;"
             "用归一化 PNG 时必须提供(可从 GT Z + 相机高度估算)",
    )
    parser.add_argument(
        "--depth-far-m",
        type=float,
        default=None,
        help="深度 PNG 最暗像素(0)对应的米数;默认 = depth-near-m 的 1.4 倍",
    )
    parser.add_argument(
        "--rgb-coord-system",
        choices=["saved", "raw"],
        default="saved",
        help="GT 中 rgb_uv 坐标系:saved(默认,640x360 depth PNG 用的缩小图坐标)或 raw(1920x1080 原始)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="打印可直接贴回 config.py 的 numpy 字面量",
    )
    args = parser.parse_args()

    if not os.path.exists(args.rgb_path):
        print(f"找不到 RGB: {args.rgb_path}", file=sys.stderr)
        sys.exit(1)

    # 1. 加载 RGB + 深度(优先 raw .npy, 退而求其次用 PNG)
    rgb = np.array(Image.open(args.rgb_path).convert("RGB"), dtype=np.uint8)
    rgb_shape = rgb.shape[:2]   # (H, W)

    using_raw = False
    if os.path.exists(args.depth_raw_path):
        depth_raw = np.load(args.depth_raw_path)
        depth_shape = depth_raw.shape[:2]
        print(f"使用原始深度: {args.depth_raw_path}  shape={depth_shape}, dtype={depth_raw.dtype}")
        print(f"  depth min={np.nanmin(depth_raw):.3f}m, max={np.nanmax(depth_raw):.3f}m")
        using_raw = True
    elif os.path.exists(args.depth_png_path):
        if args.depth_near_m is None:
            print(f"用 PNG 路径时必须指定 --depth-near-m;PNG 归一化丢失了真实米数范围",
                  file=sys.stderr)
            sys.exit(1)
        depth_png = np.array(Image.open(args.depth_png_path).convert("L"), dtype=np.uint8)
        # 把 PNG 值反归一化:PNG = 255 → near, PNG = 0 → far
        near = args.depth_near_m
        far = args.depth_far_m if args.depth_far_m else near * 1.4
        # valid_mask: PNG > 0
        depth_raw = np.where(depth_png > 0,
                             near + (depth_png.astype(np.float32) / 255.0) * (far - near),
                             np.nan)
        depth_shape = depth_raw.shape[:2]
        print(f"使用 PNG 深度: {args.depth_png_path}  near={near}m far={far}m  shape={depth_shape}")
        print(f"  ⚠ 精度差(归一化丢失了 min/max);若平台侧有 raw .npy 优先用")
        using_raw = False
    else:
        print("找不到深度文件 (--depth-raw-path 或 --depth-png-path)", file=sys.stderr)
        sys.exit(1)

    # 3. 加载 GT
    with open(args.gt_json) as f:
        gt_list = json.load(f)
    print(f"\nGT 点数: {len(gt_list)}")

    # 4. 准备 points 列表(给 PnP 用)
    points = []
    for gt in gt_list:
        u_rgb, v_rgb = gt["rgb_uv"]
        # 若 GT 用 raw 1920x1080 坐标,先缩到 saved 960x540
        if args.rgb_coord_system == "raw":
            u_rgb = u_rgb * rgb_shape[1] / 1920.0
            v_rgb = v_rgb * rgb_shape[0] / 1080.0
        u_d, v_d = rgb_uv_to_depth_uv(u_rgb, v_rgb, rgb_shape, depth_shape)
        d = get_depth_at_raw(depth_raw, u_d, v_d)
        if d is None:
            print(f"  ⚠ {gt['name']}: 像素 ({u_d:.0f}, {v_d:.0f}) 处深度无效,跳过")
            continue
        points.append({
            "uv_depth": (u_d, v_d),
            "depth_m": d,
            "xyz_gt": tuple(gt["xyz"]),
            "depth_shape": depth_shape,
            "name": gt["name"],
        })
        print(f"  {gt['name']:6s} rgb({u_rgb:.0f},{v_rgb:.0f}) → dpx({u_d:.1f},{v_d:.1f})  depth={d:.3f}m  GT={gt['xyz']}")

    if len(points) < 4:
        print(f"\n有效点 < 4(实际 {len(points)}),无法标定", file=sys.stderr)
        sys.exit(2)

    # 5. PnP 求解(cx/cy 加入搜索 + RANSAC)
    print("\n--- 求解中 (cx/cy 加入搜索 + RANSAC) ---")
    K, T, err, inlier_mask = solve_pnp(points, ransac_iters=50, outlier_thresh_m=0.10)
    print(f"内参 K:\n{K}")
    print(f"\nhead→base T:\n{T}")
    print(f"\n平均重投影误差(全点): {err*1000:.1f} mm")
    print(f"内点/外点:")
    for i, p in enumerate(points):
        tag = "IN " if inlier_mask[i] else "OUT"
        print(f"  {tag}  {p['name']:6s}  uv=({p['uv_depth'][0]:.1f},{p['uv_depth'][1]:.1f})  d={p['depth_m']:.3f}m  GT={p['xyz_gt']}")

    # 6. 打印 config.py 格式
    if args.print_config:
        print("\n" + "=" * 60)
        print("# 复制以下到 config.py (替换占位值):")
        print("=" * 60)
        print()
        print("HEAD_CAMERA_INTRINSICS = {")
        print(f'    "fx": {K[0,0]:.2f},')
        print(f'    "fy": {K[1,1]:.2f},')
        print(f'    "cx": {K[0,2]:.2f},  # depth 宽度 {depth_shape[1]} 的一半')
        print(f'    "cy": {K[1,2]:.2f},  # depth 高度 {depth_shape[0]} 的一半')
        print("}")
        print()
        print("HEAD_TO_BASE_T = np.array([")
        for row in T[:3]:
            print(f"    [{', '.join(f'{v: .6f}' for v in row)}],")
        print("    [0.0, 0.0, 0.0, 1.0],")
        print("], dtype=np.float64)")
        print()
        print("# 验证:在 4 个解算点上的重投影误差应该 < 1 cm(目标 < 5 mm)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()