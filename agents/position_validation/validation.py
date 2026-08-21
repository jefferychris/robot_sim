"""position_validation 的核心比对逻辑。

────────────────────────────────────────────────────────────────────────
责任范围:
  - 加载场景参数 JSON
  - 用 nut_picker.detector 在 RGB 上找螺母 + 箱子
  - 用 nut_picker.geometry 把每个像素反投影到 base_link XYZ
  - 与场景参数里的对象 XYZ 真值逐个比对 → 输出 mm 级欧氏误差

不在这里的职责:
  - 相机订阅(CameraHub)、overlay 保存、日志格式 — 都在 __init__.run()。
  - 检测算法、调参阈值 — 都从 nut_picker 复用。
────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from typing import Optional

import numpy as np

from agents.nut_picker import detector as nut_detector
from agents.nut_picker import geometry as nut_geometry
from agents.nut_picker import config as nut_config

from . import config as cfg


logger = logging.getLogger("position_validation")


# ══════════════════════════════════════════════════════════════════════
#  异常
# ══════════════════════════════════════════════════════════════════════
class ValidationError(RuntimeError):
    """position_validation 流程出错(场景参数缺失/检测失败/反投影失败)。"""


# ══════════════════════════════════════════════════════════════════════
#  场景参数加载
# ══════════════════════════════════════════════════════════════════════
def load_scene_params(path: Optional[str] = None) -> dict:
    """从 JSON 文件加载场景参数。

    返回:
        {
          "camera":   {"name": str, "position": [x,y,z], "orientation_rpy": [r,p,y]},
          "objects":  [{"name": str, "kind": "nut"|"box", "xyz": [x,y,z]}, ...]
        }

    异常:ValidationError — 文件不存在 / 解析失败 / 缺关键字段。
    """
    path = path or cfg.SCENE_PARAMS_PATH
    if not os.path.exists(path):
        raise ValidationError(
            f"场景参数文件不存在: {path}"
            f"(可设环境变量 POSITION_VALIDATION_SCENE_PARAMS 指向别的路径)"
        )
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"场景参数 JSON 解析失败 ({path}): {e}") from e

    # 字段校验
    if "camera" not in data:
        raise ValidationError(f"场景参数缺 camera 字段: {path}")
    cam = data["camera"]
    for key in ("position", "orientation_rpy"):
        if key not in cam:
            raise ValidationError(f"camera 缺 {key} 字段: {path}")
    if len(cam["position"]) != 3 or len(cam["orientation_rpy"]) != 3:
        raise ValidationError(f"camera.position / orientation_rpy 必须各 3 个数: {path}")

    if "objects" not in data or not data["objects"]:
        raise ValidationError(f"场景参数 objects 列表为空或缺失: {path}")
    for i, obj in enumerate(data["objects"]):
        for key in ("name", "kind", "xyz"):
            if key not in obj:
                raise ValidationError(f"objects[{i}] 缺 {key} 字段: {path}")
        if obj["kind"] not in ("nut", "box"):
            raise ValidationError(f"objects[{i}].kind 必须是 nut/box,得到 {obj['kind']}")
        if len(obj["xyz"]) != 3:
            raise ValidationError(f"objects[{i}].xyz 必须 3 个数: {path}")

    n_nuts = sum(1 for o in data["objects"] if o["kind"] == "nut")
    n_boxes = sum(1 for o in data["objects"] if o["kind"] == "box")
    logger.info(
        f"[scene] 加载 {path}: camera={cam.get('name', '?')}, "
        f"nuts={n_nuts}, boxes={n_boxes}"
    )
    return data


# ══════════════════════════════════════════════════════════════════════
#  检测 → 反投影 → 与 GT 比对
# ══════════════════════════════════════════════════════════════════════
def _match_nuts(scene_nuts: list[dict], detected_nuts: list[dict]) -> list[tuple]:
    """按 area 大→小 把检测螺母与 scene nuts 一一配对(数量必须相等)。"""
    if len(scene_nuts) != len(detected_nuts):
        raise ValidationError(
            f"螺母数量不匹配: scene={len(scene_nuts)} 检测={len(detected_nuts)}。"
            f"若场景里螺母数变了,请改 position_validation_scene.json。"
        )
    # detector 输出的 nuts 已按 big/medium/small 排好(见 detector.detect_nuts),
    # 直接 zip 即可。
    return list(zip(scene_nuts, detected_nuts))


def _project_pixel(
    u_rgb: float,
    v_rgb: float,
    rgb_shape: tuple,
    depth: np.ndarray,
) -> Optional[tuple]:
    """RGB 像素 → base XYZ,完全走 nut_picker.geometry 的标定(K + T)。"""
    return nut_geometry.pixel_to_base(
        u_rgb, v_rgb,
        rgb_shape=rgb_shape,
        depth=depth,
        intrinsics=cfg.HEAD_CAMERA_INTRINSICS,
        head_to_base_T=cfg.HEAD_TO_BASE_T,
        table_z_fallback_m=cfg.TABLE_Z_FALLBACK_M,
    )


def _diff_m(a: tuple, b: tuple) -> float:
    """两个 (x, y, z) 的欧氏距离,返回米。"""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def run_validation(
    rgb: np.ndarray,
    depth: np.ndarray,
    scene: dict,
) -> list[dict]:
    """核心入口:在 (rgb, depth) 上做检测,把每个对象的反投影 XYZ 与 scene GT 比。

    list 元素 dict:
        - name:        scene 里的对象名
        - kind:        "nut" | "box"
        - gt_xyz:      scene 里的真值 XYZ (米)
        - pred_xyz:    depth+检测解算出的 XYZ(米),失败时 None
        - error_m:     欧氏误差(米),失败时 None
        - error_mm:    同上转 mm
        - status:      "PASS" | "FAIL" | "EXCELLENT" | "MISS"
        - centroid_px: (u, v) 像素(便于 overlay),pred 失败时 None

    异常:ValidationError — 检测失败(走 nut_picker 的 DetectionError)。
    """
    rgb_shape = rgb.shape[:2]   # (H, W)
    H, W = rgb_shape

    # ── 1. 检测 ────────────────────────────────────────────────────
    # 包一层:detector 抛 DetectionError → 这里统一为 ValidationError
    try:
        nuts = nut_detector.detect_nuts(rgb)
        box = nut_detector.detect_box(rgb)
    except nut_detector.DetectionError as e:
        raise ValidationError(f"检测失败: {e}") from e

    # ── 2. 分类 scene 对象 ─────────────────────────────────────────
    scene_nuts = [o for o in scene["objects"] if o["kind"] == "nut"]
    scene_boxes = [o for o in scene["objects"] if o["kind"] == "box"]
    if not scene_boxes:
        raise ValidationError("场景参数里没有任何 kind=='box' 的对象")

    # ── 3. 比对螺母 ───────────────────────────────────────────────
    results: list[dict] = []
    try:
        nut_pairs = _match_nuts(scene_nuts, nuts)
    except ValidationError:
        # 数量不匹配时,把已有的也记下,再 raise 出去
        for obj in scene_nuts:
            results.append({
                "name": obj["name"], "kind": "nut",
                "gt_xyz": tuple(obj["xyz"]),
                "pred_xyz": None, "error_m": None, "error_mm": None,
                "status": "MISS", "centroid_px": None,
            })
        raise

    for obj, det in nut_pairs:
        u, v = det["centroid_px"]
        pred = _project_pixel(u, v, rgb_shape, depth)
        if pred is None:
            results.append({
                "name": obj["name"], "kind": "nut",
                "gt_xyz": tuple(obj["xyz"]),
                "pred_xyz": None, "error_m": None, "error_mm": None,
                "status": "MISS", "centroid_px": (u, v),
            })
            continue
        err = _diff_m(pred, tuple(obj["xyz"]))
        results.append({
            "name": obj["name"], "kind": "nut",
            "gt_xyz": tuple(obj["xyz"]),
            "pred_xyz": pred,
            "error_m": err, "error_mm": err * 1000.0,
            "status": _status(err * 1000.0),
            "centroid_px": (u, v),
        })

    # ── 4. 比对箱子(只看箱子中心 1 个点,与 scene 里第一个 box 对齐) ──
    bx, by, bw, bh = box["bbox"]
    cx_px = bx + bw / 2.0
    cy_px = by + bh / 2.0
    box_gt = scene_boxes[0]   # 只取第一个 box
    pred = _project_pixel(cx_px, cy_px, rgb_shape, depth)
    if pred is None:
        results.append({
            "name": box_gt["name"], "kind": "box",
            "gt_xyz": tuple(box_gt["xyz"]),
            "pred_xyz": None, "error_m": None, "error_mm": None,
            "status": "MISS", "centroid_px": (cx_px, cy_px),
        })
    else:
        err = _diff_m(pred, tuple(box_gt["xyz"]))
        results.append({
            "name": box_gt["name"], "kind": "box",
            "gt_xyz": tuple(box_gt["xyz"]),
            "pred_xyz": pred,
            "error_m": err, "error_mm": err * 1000.0,
            "status": _status(err * 1000.0),
            "centroid_px": (cx_px, cy_px),
        })

    return results


def _status(error_mm: float) -> str:
    if error_mm < cfg.EXCELLENT_MM:
        return "EXCELLENT"
    if error_mm < cfg.OK_MM:
        return "PASS"
    return "FAIL"


# ══════════════════════════════════════════════════════════════════════
#  日志 + 统计
# ══════════════════════════════════════════════════════════════════════
def summarize(results: list[dict]) -> dict:
    """汇总误差统计(只算有 pred 的项)。"""
    valid = [r for r in results if r["pred_xyz"] is not None]
    n_valid = len(valid)
    n_total = len(results)
    if n_valid == 0:
        return {
            "n_total": n_total, "n_valid": 0, "n_miss": n_total,
            "mean_mm": None, "max_mm": None, "min_mm": None,
            "pass_count": 0, "fail_count": 0, "excellent_count": 0,
        }
    errs = [r["error_mm"] for r in valid]
    return {
        "n_total": n_total, "n_valid": n_valid, "n_miss": n_total - n_valid,
        "mean_mm": sum(errs) / n_valid,
        "max_mm": max(errs),
        "min_mm": min(errs),
        "pass_count": sum(1 for r in valid if r["status"] in ("PASS", "EXCELLENT")),
        "fail_count": sum(1 for r in valid if r["status"] == "FAIL"),
        "excellent_count": sum(1 for r in valid if r["status"] == "EXCELLENT"),
    }


def log_results(results: list[dict]) -> None:
    """把每个对象的 GT / pred / 误差打到日志。"""
    for r in results:
        if r["pred_xyz"] is None:
            logger.warning(
                f"[{r['status']}] {r['kind']:3s} {r['name']:>10s}  "
                f"gt={tuple(round(v, 3) for v in r['gt_xyz'])}  pred=MISS"
            )
            continue
        logger.info(
            f"[{r['status']:>9s}] {r['kind']:3s} {r['name']:>10s}  "
            f"gt={tuple(round(v, 3) for v in r['gt_xyz'])}  "
            f"pred={tuple(round(v, 3) for v in r['pred_xyz'])}  "
            f"err={r['error_mm']:6.1f} mm"
        )
    s = summarize(results)
    logger.info(
        f"[summary] {s['n_valid']}/{s['n_total']} 有解;"
        f" 平均={s['mean_mm']:.1f}mm 最大={s['max_mm']:.1f}mm 最小={s['min_mm']:.1f}mm;"
        f" PASS={s['pass_count']} FAIL={s['fail_count']} MISS={s['n_miss']}"
    )