"""position_validation 的离线重算脚本。

────────────────────────────────────────────────────────────────────────
为什么需要这个脚本?
  `agents/position_validation/__init__.run()` 是「在线」流程:订阅相机 → 等首帧 → 检测 → 比对。
  但用户复盘 calibration 时,常用的是已经跑过的 saved 文件:
    - camera_frames/head_rgb.png            (960x540, RGB, PNG 无损)
    - camera_frames/head_depth_raw.npy      (640x360, float32, **米数**)

  这个脚本就是离线版的 run():直接读这两份**原始**文件,把 pipeline 跑一遍并打印。
  注意位置(3D XYZ)是从 raw depth 反投影算的 ——
  **RGB 只用来定位「哪一个像素」是螺母/箱**,不参与 3D 坐标计算。

数据流(显式打印):
  1. RGB (960x540) ── detector ──> 像素 (u, v)
  2. 像素 (u, v) ── 按比例缩放到 depth 分辨率 ──> 像素 (u_d, v_d)
  3. depth[u_d, v_d] (raw 米数) ── 针孔反投影 ──> 相机坐标 (x_c, y_c, z)
  4. (x_c, y_c, z) ── head→base T ──> base_link XYZ

用法:
    python -m agents.position_validation.validation_offline
    python -m agents.position_validation.validation_offline \
        --rgb-path camera_frames/head_rgb.png \
        --depth-raw-path camera_frames/head_depth_raw.npy \
        --scene-params calibration_runs/position_validation_scene.json
────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image


# 让 `python -m agents.position_validation.validation_offline` 能找到包内模块
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents.position_validation import config as cfg
from agents.position_validation import validation


logger = logging.getLogger("position_validation.offline")


# ══════════════════════════════════════════════════════════════════════
#  数据加载(显式区分 RGB / raw depth)
# ══════════════════════════════════════════════════════════════════════
def load_raw_rgb(path: str) -> np.ndarray:
    """从 PNG 读 RGB(960x540 uint8)。PNG 无损 = 原始 RGB 像素。

    注意:RGB **不参与** 3D 坐标计算,只用来定位目标在「哪个像素」。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 RGB: {path}")
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    logger.info(
        f"[load] RGB ← {path}  shape={rgb.shape} dtype={rgb.dtype}"
        f"(仅用于找目标像素)"
    )
    return rgb


def load_raw_depth(path: str) -> np.ndarray:
    """从 .npy 读 raw depth(640x360 float32,单位**米**)。

    这是 3D 位置计算的**唯一数据源**(针孔反投影的 z 值)。
    不同于 head_depth.png(归一化到 0-255 已丢米数),raw .npy 是原始 float32。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 raw depth: {path}")
    depth = np.load(path)
    if depth.dtype != np.float32:
        logger.warning(
            f"[load] raw depth dtype={depth.dtype} 不是 float32,强转"
        )
        depth = depth.astype(np.float32)
    valid = np.isfinite(depth)
    logger.info(
        f"[load] raw depth ← {path}  shape={depth.shape} dtype={depth.dtype}"
        f"(用于 3D 反投影)"
    )
    logger.info(
        f"  有效像素 {valid.sum()}/{depth.size} "
        f"min={depth[valid].min():.3f}m max={depth[valid].max():.3f}m"
    )
    return depth


# ══════════════════════════════════════════════════════════════════════
#  数据流打印(让用户看清楚 RGB → 像素 → depth → 3D)
# ══════════════════════════════════════════════════════════════════════
def print_data_flow(rgb: np.ndarray, depth: np.ndarray,
                    rgb_uv: tuple, label: str) -> None:
    """把单个目标(RGB 像素 → depth 米数 → 3D)走过的每一步打到日志。

    读 raw 数据并打到屏幕上,验证「3D 来自 raw depth」这件事。
    """
    from agents.nut_picker import geometry as geom
    u_rgb, v_rgb = rgb_uv
    H_rgb, W_rgb = rgb.shape[:2]
    H_d, W_d = depth.shape[:2]

    # RGB 像素 → depth 像素(按比例缩放)
    u_d = u_rgb * W_d / W_rgb
    v_d = v_rgb * H_d / H_rgb

    # 取 depth(邻域中位数,降噪 — 用 calibrate.py 同款)
    ui, vi = int(round(u_d)), int(round(v_d))
    d_val = None
    if 0 <= ui < W_d and 0 <= vi < H_d:
        # 3x3 邻域中位数
        vals = []
        for dv in range(-1, 2):
            for du in range(-1, 2):
                uu, vv = ui + du, vi + dv
                if 0 <= uu < W_d and 0 <= vv < H_d:
                    v = float(depth[vv, uu])
                    if np.isfinite(v) and v > 0:
                        vals.append(v)
        if vals:
            d_val = float(np.median(vals))

    # 针孔反投影
    p_base = None
    if d_val is not None:
        K = cfg.HEAD_CAMERA_INTRINSICS
        x_cam = (u_d - K["cx"]) * d_val / K["fx"]
        y_cam = (v_d - K["cy"]) * d_val / K["fy"]
        p_cam = np.array([x_cam, y_cam, d_val, 1.0])
        p_base = cfg.HEAD_TO_BASE_T @ p_cam
        p_base = (float(p_base[0]), float(p_base[1]), float(p_base[2]))

    logger.info(
        f"  {label}: rgb_px=({u_rgb:.0f},{v_rgb:.0f})  "
        f"depth_px=({u_d:.1f},{v_d:.1f})  "
        f"depth={d_val:.4f}m  "
        f"base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})"
        if d_val is not None and p_base is not None else
        f"  {label}: rgb_px=({u_rgb:.0f},{v_rgb:.0f})  depth_px=({u_d:.1f},{v_d:.1f})  "
        f"depth=INVALID"
    )


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="position_validation 离线重算:读 raw RGB + raw depth,跑 pipeline"
    )
    SAVE_DIR = "camera_frames"
    p.add_argument(
        "--rgb-path",
        default=os.path.join(SAVE_DIR, "head_rgb.png"),
        help="原始 RGB PNG 路径(用于目标像素定位)",
    )
    p.add_argument(
        "--depth-raw-path",
        default=os.path.join(SAVE_DIR, "head_depth_raw.npy"),
        help="raw depth .npy 路径(**米数**,用于 3D 反投影)",
    )
    p.add_argument(
        "--scene-params",
        default=cfg.SCENE_PARAMS_PATH,
        help="场景参数 JSON 路径",
    )
    p.add_argument(
        "--verbose-flow", action="store_true",
        help="打印每个目标 RGB→depth→3D 的完整数据流",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── 1. 读原始数据(显式区分 RGB vs raw depth) ─────────────────
    logger.info("=" * 70)
    logger.info("[step 1/4] 加载原始数据")
    logger.info("=" * 70)
    rgb = load_raw_rgb(args.rgb_path)
    depth = load_raw_depth(args.depth_raw_path)
    logger.info(
        f"  分辨率差异: RGB={rgb.shape[1]}x{rgb.shape[0]}"
        f"  raw_depth={depth.shape[1]}x{depth.shape[0]} "
        f"(geometry 内部按比例缩放)"
    )

    # ── 2. 加载场景参数 ─────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("[step 2/4] 加载场景参数")
    logger.info("=" * 70)
    scene = validation.load_scene_params(args.scene_params)

    # ── 3. 检测 + 反投影 + 比对 ──────────────────────────────────
    logger.info("=" * 70)
    logger.info("[step 3/4] 检测(RGB)→ 反投影(raw depth)→ 比对 GT")
    logger.info("=" * 70)
    logger.info("  → RGB 只用来找目标像素,不参与 3D 计算")
    logger.info("  → raw depth 提供 z 值(米)→ 针孔反投影 → head→base T → base_link XYZ")
    results = validation.run_validation(rgb, depth, scene)

    # 可选:逐个目标的完整数据流
    if args.verbose_flow:
        logger.info("--- 详细数据流 ---")
        for r in results:
            if r["centroid_px"] is not None:
                print_data_flow(rgb, depth, r["centroid_px"], r["name"])

    # ── 4. 输出 ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("[step 4/4] 比对结果")
    logger.info("=" * 70)
    validation.log_results(results)


if __name__ == "__main__":
    main()