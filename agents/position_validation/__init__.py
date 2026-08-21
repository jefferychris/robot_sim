"""position_validation agent 入口。

────────────────────────────────────────────────────────────────────────
非 LLM agent(plain script 风格,跟 camera_demo / nut_picker 一致)。
平台以 `python3 -u main.py position_validation` 启动,加载本子包并调 run()。

流程:
  1. 加载场景参数(相机位姿 + 螺母/收纳盒 XYZ 真值,JSON 文件)
  2. CameraHub 订阅 head_rgb + head_depth,等首帧
  3. validation.run_validation:detector 找螺母+箱 → geometry 反投影到 base XYZ
  4. 与场景参数 GT 逐个比对 → 误差(欧氏,mm)
  5. 打印每个对象 + 总体统计 + 保存 overlay PNG

环境变量:
  POSITION_VALIDATION_SCENE_PARAMS=path/to/scene.json
  POSITION_VALIDATION_SAVE_OVERLAY=0  不保存 overlay
────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import time

import numpy as np

from drivers.camera import CameraHub

from . import config
from . import validation


__all__ = ["run"]


logger = logging.getLogger("position_validation")


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


def _save_overlay(rgb: np.ndarray, results: list[dict]) -> str | None:
    """在 RGB 上画检测框 + GT 标注 + 误差文字 → 保存 PNG。"""
    import cv2
    os.makedirs(config.OVERLAY_DIR, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb.shape[2] == 3 else rgb

    # 兼容 RGB→BGR(画图统一走 BGR 通道)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    colors_bbox = {
        "EXCELLENT": (0, 200, 0),     # 绿
        "PASS":      (0, 200, 200),   # 黄
        "FAIL":      (0, 0, 220),     # 红
        "MISS":      (180, 180, 180), # 灰
    }

    for r in results:
        if r["centroid_px"] is None:
            continue
        cx, cy = [int(round(v)) for v in r["centroid_px"]]
        color = colors_bbox.get(r["status"], (200, 200, 200))
        # 画中心十字
        cv2.drawMarker(bgr, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
        # 标签(对象名 + 误差)
        if r["pred_xyz"] is None:
            label = f"{r['name']} MISS"
        else:
            label = f"{r['name']} {r['error_mm']:.0f}mm"
        cv2.putText(
            bgr, label, (cx + 12, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    # 顶部摘要条
    s = validation.summarize(results)
    if s["mean_mm"] is not None:
        head = (
            f"n={s['n_valid']}/{s['n_total']}  "
            f"mean={s['mean_mm']:.1f}mm  max={s['max_mm']:.1f}mm  "
            f"PASS={s['pass_count']} FAIL={s['fail_count']} MISS={s['n_miss']}"
        )
    else:
        head = f"n={s['n_valid']}/{s['n_total']} (no valid predictions)"
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(bgr, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(config.OVERLAY_DIR, f"{config.OVERLAY_PREFIX}_{ts}.png")
    cv2.imwrite(out, bgr)
    return out


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    save_overlay = config.SAVE_OVERLAY
    logger.info(
        f"[position_validation] 启动: SAVE_OVERLAY={save_overlay} "
        f"SCENE_PARAMS={config.SCENE_PARAMS_PATH}"
    )

    # ── 1. 加载场景参数 ───────────────────────────────────────────
    try:
        scene = validation.load_scene_params(config.SCENE_PARAMS_PATH)
    except validation.ValidationError as e:
        logger.error(f"[position_validation] {e}")
        return None

    # ── 2. 相机订阅 ───────────────────────────────────────────────
    hub = CameraHub(config.CAMERAS, node_name="position_validation")
    logger.info(f"[position_validation] 已订阅 {len(config.CAMERAS)} 路相机: {list(config.CAMERAS)}")

    frames = _wait_first_frames(hub, config.WAIT_TIMEOUT)
    rgb_frame = frames.get("head_rgb")
    depth_frame = frames.get("head_depth")
    if rgb_frame is None or depth_frame is None:
        logger.error(
            f"[position_validation] 未收到 head_rgb 或 head_depth,"
            f"frames={list(frames.keys())}"
        )
        hub.shutdown()
        return None
    rgb = rgb_frame["array"]
    depth = depth_frame["array"]
    logger.info(
        f"[position_validation] rgb.shape={rgb.shape}  depth.shape={depth.shape}"
    )

    # ── 3. 检测 + 比对 ────────────────────────────────────────────
    try:
        results = validation.run_validation(rgb, depth, scene)
    except validation.ValidationError as e:
        logger.error(f"[position_validation] 比对失败: {e}")
        hub.shutdown()
        return None

    # ── 4. 日志 + 统计 ────────────────────────────────────────────
    validation.log_results(results)
    summary = validation.summarize(results)

    # ── 5. overlay PNG ────────────────────────────────────────────
    if save_overlay:
        try:
            out = _save_overlay(rgb, results)
            if out:
                logger.info(f"[position_validation] overlay 已保存: {out}")
        except Exception as e:
            logger.warning(f"[position_validation] overlay 保存失败: {e}")

    hub.shutdown()
    logger.info("[position_validation] 退出")
    return {"results": results, "summary": summary}