"""像素坐标 → 机器人 base 坐标。

────────────────────────────────────────────────────────────────────────
管线(从 RGB 像素到 base_link):
    (u_rgb, v_rgb) in head_rgb   [1920x1080 或类似尺寸]
        ↓  按比例缩放 (rgb→depth 分辨率)
    (u_d, v_d)  in head_depth   [640x360]
        ↓  查 depth[u_d, v_d] 得深度 z(米,float32)
        ↓  针孔反投影(相机内参 fx/fy/cx/cy)
    (x_cam, y_cam, z_cam)
        ↓  4x4 齐次变换(head→base)
    (x_base, y_base, z_base)

降级路径:
    当 depth[u_d, v_d] 是 inf / nan / 缺数据时,改用 TABLE_Z_FALLBACK_M 作 Z。
    此时 XY 仍走针孔公式(隐含假设相机帧≈base 帧,README 注明)。
────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("nut_picker.geometry")


def depth_at(
    depth: np.ndarray,
    u_d: float,
    v_d: float,
) -> Optional[float]:
    """从深度图里取最近邻像素的深度(米)。无效值返回 None。

    depth: (H, W) float32,单位米,inf/nan 表示无效。
    """
    h, w = depth.shape[:2]
    ui = int(round(u_d))
    vi = int(round(v_d))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    val = float(depth[vi, ui])
    if not np.isfinite(val):
        return None
    return val


def pixel_to_base(
    u_rgb: float,
    v_rgb: float,
    rgb_shape: Tuple[int, int],
    depth: Optional[np.ndarray],
    intrinsics: dict,
    head_to_base_T: np.ndarray,
    table_z_fallback_m: float = 0.02,
) -> Optional[Tuple[float, float, float]]:
    """像素 → base_link XYZ。

    参数:
        u_rgb, v_rgb:           RGB 像素坐标
        rgb_shape:              (H, W) RGB 图像尺寸
        depth:                  (H_d, W_d) float32 深度图(米),可为 None
        intrinsics:             {"fx","fy","cx","cy"}
        head_to_base_T:         4×4 numpy 矩阵(camera→base)
        table_z_fallback_m:     深度缺失时使用的 Z

    返回:(x, y, z) 或 None(若 RGB 像素越界)。
    """
    H, W = rgb_shape
    H_d, W_d = depth.shape[:2] if depth is not None else (None, None)

    # RGB → depth 坐标缩放
    sx = W_d / W if W_d else 1.0
    sy = H_d / H if H_d else 1.0
    u_d = u_rgb * sx
    v_d = v_rgb * sy

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # 深度(降级:无效或缺失就用 fallback)
    z = None
    if depth is not None:
        z = depth_at(depth, u_d, v_d)
    if z is None:
        z = table_z_fallback_m
        logger.debug(
            f"[geom] depth 缺失/无效,使用 fallback Z={z}m at ({u_rgb:.1f},{v_rgb:.1f})"
        )

    # 针孔反投影(camera frame)
    x_cam = (u_d - cx) * z / fx
    y_cam = (v_d - cy) * z / fy
    p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)

    # camera → base
    p_base = head_to_base_T @ p_cam
    return float(p_base[0]), float(p_base[1]), float(p_base[2])