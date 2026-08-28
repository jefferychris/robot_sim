"""nut_picker agent 入口。

────────────────────────────────────────────────────────────────────────
非 LLM agent(plain script 风格,跟 camera_demo / arm_hand_demo 一致)。
平台以 `python3 -u main.py` 启动,加载本子包并调 run()。

流程:
  1. CameraHub 订阅 head_rgb + head_depth,等首帧
  2. detector.detect_nuts / detect_box 找 3 个螺母(按面积排大/中/小)+ 3 个 cell
  3. geometry.pixel_to_base 把每个像素反投影到 base_link XYZ
  4. motion.PickPlaceRunner 顺序执行 big→cell0、medium→cell1、small→cell2

环境变量:
  NUT_PICKER_DRY_RUN=1   只检测+打印动作序列,不实际驱动机械臂
  NUT_PICKER_SAVE_OVERLAY=0  不保存带标注的 PNG(默认保存到 camera_frames/)

────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import time

import numpy as np

from drivers.camera import CameraHub
from rabo_robocap import LinkerArmA7, LinkerHandO6Right

from . import config
from . import detector
from . import geometry
from . import motion as motion_module


__all__ = ["run"]


logger = logging.getLogger("nut_picker")


def _wait_first_frames(hub: CameraHub, timeout: float) -> dict:
    """等到所有相机都收到至少一帧,返回 get_all() 快照。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        frames = hub.get_all()
        missing = [n for n in config.CAMERAS if n not in frames]
        if not missing:
            return frames
        time.sleep(0.2)
    return hub.get_all()


def _save_overlay(rgb: np.ndarray, nuts: list[dict], box: dict) -> None:
    """把检测结果画到 RGB 上,保存为 PNG(便于人工调参)。"""
    import cv2
    os.makedirs(config.OVERLAY_DIR, exist_ok=True)
    bgr = detector.draw_overlay(rgb, nuts, box)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(config.OVERLAY_DIR, f"nut_picker_overlay_{ts}.png")
    cv2.imwrite(out, bgr)
    logger.info(f"[nut_picker] 可视化已保存: {out}")


def _save_debug_overlay(rgb: np.ndarray, *, note: str = "",
                        nuts: list[dict] | None = None,
                        box: dict | None = None,
                        prefix: str = "debug") -> str | None:
    """即使 nuts 或 box 为空也保存调试图(显示 mask/候选轮廓等)。返回路径。"""
    import cv2
    n_info = detector.inspect_nuts(rgb)
    b_info = detector.inspect_box(rgb)
    panel = detector.draw_debug_overlay(
        rgb,
        nut_mask=n_info["mask"],
        box_mask=b_info["mask"],
        raw_contours=n_info["raw_contours"],
        kept_contours=n_info["kept_contours"],
        box_candidates=b_info["box_candidates"],
        nuts=nuts,
        box=box,
        note=note,
    )
    os.makedirs(config.OVERLAY_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(config.OVERLAY_DIR, f"nut_picker_{prefix}_{ts}.png")
    cv2.imwrite(out, panel)
    logger.info(f"[nut_picker] 调试图已保存: {out}")
    return out


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    dry_run = config.DRY_RUN
    save_overlay = config.SAVE_OVERLAY
    logger.info(
        f"[nut_picker] 启动:DRY_RUN={dry_run} SAVE_OVERLAY={save_overlay} "
        f"ARM_MODE={config.ARM_MODE}"
    )

    # ── 1. 相机 ────────────────────────────────────────────────────
    hub = CameraHub(config.CAMERAS, node_name="nut_picker")
    logger.info(f"[nut_picker] 已订阅 {len(config.CAMERAS)} 路相机: {list(config.CAMERAS)}")

    frames = _wait_first_frames(hub, config.WAIT_TIMEOUT)
    rgb_frame = frames.get("head_rgb")
    depth_frame = frames.get("head_depth")
    if rgb_frame is None or depth_frame is None:
        logger.error(
            f"[nut_picker] 未收到 head_rgb 或 head_depth,frames={list(frames.keys())}"
        )
        hub.shutdown()
        return None
    rgb = rgb_frame["array"]
    depth = depth_frame["array"]
    rgb_shape = (rgb_frame["height"], rgb_frame["width"])  # (H, W)
    logger.info(f"[nut_picker] rgb.shape={rgb.shape}  depth.shape={depth.shape}")

    # ── 2. 检测 ────────────────────────────────────────────────────
    nuts = None
    box = None
    try:
        nuts = detector.detect_nuts(rgb)
        box = detector.detect_box(rgb)
    except detector.DetectionError as e:
        logger.error(f"[nut_picker] 检测失败: {e}")
        # 即使失败也输出可视调试图(显示 mask/候选轮廓等),方便调参
        _save_debug_overlay(rgb, note=f"DETECTION FAILED: {e}",
                            nuts=nuts, box=box, prefix="fail")
        hub.shutdown()
        return None

    # ── 3. 可视化(可选)───────────────────────────────────────────────
    if save_overlay:
        _save_overlay(rgb, nuts, box)

    # ── 4. 反投影到 base 坐标 ────────────────────────────────────────
    pairs = list(zip(nuts, box["cells"]))   # 按 (big, cell0), (medium, cell1), (small, cell2)
    targets = []  # [(nut_label, nut_xyz, cell_idx, cell_xyz), ...]
    for nut, cell in pairs:
        n_xyz = geometry.pixel_to_base(
            *nut["centroid_px"], rgb_shape=rgb_shape, depth=depth,
            intrinsics=config.HEAD_CAMERA_INTRINSICS,
            head_to_base_T=config.HEAD_TO_BASE_T,
            table_z_fallback_m=config.TABLE_Z_FALLBACK_M, depth_scale=config.DEPTH_SCALE_CORRECTION,
            fixed_z=config.TABLE_Z_M if config.USE_FIXED_TABLE_Z else None,
        )
        c_xyz = geometry.pixel_to_base(
            *cell["centroid_px"], rgb_shape=rgb_shape, depth=depth,
            intrinsics=config.HEAD_CAMERA_INTRINSICS,
            head_to_base_T=config.HEAD_TO_BASE_T,
            table_z_fallback_m=config.TABLE_Z_FALLBACK_M, depth_scale=config.DEPTH_SCALE_CORRECTION,
            fixed_z=config.TABLE_Z_M if config.USE_FIXED_TABLE_Z else None,
        )
        if n_xyz is None or c_xyz is None:
            logger.error(f"[nut_picker] 反投影失败: nut={nut['label']} cell={cell['index']}")
            hub.shutdown()
            return None
        targets.append((nut["label"], n_xyz, cell["index"], c_xyz))
        logger.info(
            f"[plan] {nut['label']} @ {tuple(round(v, 3) for v in n_xyz)}"
            f"  →  cell{cell['index']} @ {tuple(round(v, 3) for v in c_xyz)}"
        )

    # ── 5. 执行抓放 ─────────────────────────────────────────────────
    arm = LinkerArmA7(robot_id=config.RIGHT_ARM_ID, mode=config.ARM_MODE)
    hand = LinkerHandO6Right(robot_id=config.RIGHT_HAND_ID, mode=config.HAND_MODE)
    runner = motion_module.PickPlaceRunner(arm, hand, config)

    try:
        runner.home()
        for label, n_xyz, c_idx, c_xyz in targets:
            if dry_run:
                logger.info(
                    f"[dry-run] pick {label} @ {n_xyz} → cell{c_idx} @ {c_xyz} (skip motion)"
                )
                continue
            logger.info(f"[exec] pick {label} @ {n_xyz} → cell{c_idx} @ {c_xyz}")
            # 直接用 depth 反投影出的 XYZ(motion 内部加 APPROACH/GRASP/PLACE 偏移)
            nx, ny, nz = n_xyz
            cx, cy, cz = c_xyz
            ok_pick = runner.pick(nx, ny, nz)
            if not ok_pick:
                logger.warning(f"[exec] pick {label} 失败,跳过 place")
                continue
            ok_place = runner.place(cx, cy, cz)
            if not ok_place:
                logger.warning(f"[exec] place {label} 失败")
        runner.home()
    except motion_module.MotionError as e:
        logger.error(f"[nut_picker] 运动失败: {e}")
    finally:
        runner.shutdown()
        hub.shutdown()

    logger.info("[nut_picker] 退出")
    return targets