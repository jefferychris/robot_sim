"""position_validation 双模式验证脚本。

────────────────────────────────────────────────────────────────────────
为什么需要这个脚本?
  calibrate.py 标定出来的 HEAD_CAMERA_INTRINSICS={fx:656.84, fy:862.11}(非方像素、
  fy > fx 不少)跟「场景真值」{fx:fy:554.3, 假设方像素、针孔、h_fov=60°}对不上。
  这个脚本用**场景推导的 intrinsics** 跑两次 pipeline,看误差来源主要是:

    Mode A:scene intrinsics + calibrated HEAD_TO_BASE_T
            (只换 K,外参用现有标定 → 看 intrinsic 误差占多少)
    Mode B:scene intrinsics + scene camera mount pose(假设 obzg_33 == base_link)
            (K + T 都用场景 → 假设 mount_link 就是 base_link,看全场景误差)

两者都跟 scene 里的对象 GT 比对,直观对比两组 mm 误差。

用法:
    python -m agents.position_validation.validate_scene_intrinsics
    python -m agents.position_validation.validate_scene_intrinsics \
        --scene-params calibration_runs/position_validation_scene.json
────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
from PIL import Image


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents.position_validation import config as cfg
from agents.position_validation import validation
from agents.nut_picker import detector as nut_detector


logger = logging.getLogger("position_validation.scene_intrinsics")


# ══════════════════════════════════════════════════════════════════════
#  几何工具
# ══════════════════════════════════════════════════════════════════════
def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY(rad) → 3x3 旋转矩阵,R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(xyz: list, rpy: list) -> np.ndarray:
    """xyz + rpy → 4x4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


# ══════════════════════════════════════════════════════════════════════
#  双模式验证
# ══════════════════════════════════════════════════════════════════════
def detect_targets(rgb: np.ndarray) -> tuple:
    """返回 (nuts, box, depth_to_use) — 不依赖 scene JSON,只做检测。"""
    nuts = nut_detector.detect_nuts(rgb)
    box = nut_detector.detect_box(rgb)
    return nuts, box


def evaluate_mode(
    name: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict,
    head_to_base_T: np.ndarray,
    nuts: list,
    box: dict,
    scene_objects: list[dict],
) -> list[dict]:
    """单模式评估:用给定的 K + T 投影每个目标的像素 → base XYZ,与 GT 比对。"""
    rgb_shape = rgb.shape[:2]

    # 按 area 大→小 与 scene nuts 配对
    scene_nuts = [o for o in scene_objects if o["kind"] == "nut"]
    scene_boxes = [o for o in scene_objects if o["kind"] == "box"]
    if len(scene_nuts) != len(nuts):
        logger.warning(
            f"[{name}] 螺母数不匹配: scene={len(scene_nuts)} 检测={len(nuts)}"
        )

    results: list[dict] = []

    # 螺母
    for i, nut in enumerate(nuts):
        if i >= len(scene_nuts):
            break
        u, v = nut["centroid_px"]
        pred = validation.project_pixel_with_K_T(
            u, v, rgb_shape, depth,
            intrinsics=intrinsics,
            head_to_base_T=head_to_base_T,
        )
        gt = tuple(scene_nuts[i]["xyz"])
        if pred is None:
            results.append({
                "name": scene_nuts[i]["name"], "kind": "nut",
                "gt_xyz": gt, "pred_xyz": None,
                "error_m": None, "error_mm": None, "status": "MISS",
                "centroid_px": (u, v),
            })
            continue
        err = float(np.linalg.norm(np.array(pred) - np.array(gt)))
        results.append({
            "name": scene_nuts[i]["name"], "kind": "nut",
            "gt_xyz": gt, "pred_xyz": pred,
            "error_m": err, "error_mm": err * 1000.0,
            "status": validation._status(err * 1000.0),
            "centroid_px": (u, v),
        })

    # 箱子中心
    if scene_boxes:
        bx, by, bw, bh = box["bbox"]
        cx_px = bx + bw / 2.0
        cy_px = by + bh / 2.0
        pred = validation.project_pixel_with_K_T(
            cx_px, cy_px, rgb_shape, depth,
            intrinsics=intrinsics,
            head_to_base_T=head_to_base_T,
        )
        gt = tuple(scene_boxes[0]["xyz"])
        if pred is None:
            results.append({
                "name": scene_boxes[0]["name"], "kind": "box",
                "gt_xyz": gt, "pred_xyz": None,
                "error_m": None, "error_mm": None, "status": "MISS",
                "centroid_px": (cx_px, cy_px),
            })
        else:
            err = float(np.linalg.norm(np.array(pred) - np.array(gt)))
            results.append({
                "name": scene_boxes[0]["name"], "kind": "box",
                "gt_xyz": gt, "pred_xyz": pred,
                "error_m": err, "error_mm": err * 1000.0,
                "status": validation._status(err * 1000.0),
                "centroid_px": (cx_px, cy_px),
            })

    return results


def log_mode(mode_name: str, results: list[dict]) -> None:
    """打印单模式结果。"""
    logger.info(f"  [{mode_name}]")
    for r in results:
        if r["pred_xyz"] is None:
            logger.info(
                f"    [{r['status']:>9s}] {r['kind']:3s} {r['name']:>10s}  "
                f"gt={tuple(round(v, 3) for v in r['gt_xyz'])}  pred=MISS"
            )
            continue
        logger.info(
            f"    [{r['status']:>9s}] {r['kind']:3s} {r['name']:>10s}  "
            f"gt={tuple(round(v, 3) for v in r['gt_xyz'])}  "
            f"pred={tuple(round(v, 3) for v in r['pred_xyz'])}  "
            f"err={r['error_mm']:6.1f} mm"
        )
    s = validation.summarize(results)
    if s["mean_mm"] is not None:
        logger.info(
            f"    [summary] mean={s['mean_mm']:.1f}mm max={s['max_mm']:.1f}mm "
            f"min={s['min_mm']:.1f}mm PASS={s['pass_count']} FAIL={s['fail_count']}"
        )
    else:
        logger.info("    [summary] no valid predictions")


# ══════════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="用场景 intrinsics 跑两种模式的 position_validation"
    )
    p.add_argument(
        "--rgb-path", default="camera_frames/head_rgb.png",
        help="原始 RGB PNG 路径",
    )
    p.add_argument(
        "--depth-raw-path", default="camera_frames/head_depth_raw.npy",
        help="raw depth .npy 路径",
    )
    p.add_argument(
        "--scene-params", default=cfg.SCENE_PARAMS_PATH,
        help="场景参数 JSON(必须含 camera.intrinsics + camera.mount_pose)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── 1. 数据 ─────────────────────────────────────────────────
    rgb = np.array(Image.open(args.rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(args.depth_raw_path).astype(np.float32)
    logger.info(
        f"[load] RGB={rgb.shape}  raw_depth={depth.shape} "
        f"range=[{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]m"
    )

    scene = validation.load_scene_params(args.scene_params)
    if "intrinsics" not in scene["camera"] or "mount_pose" not in scene["camera"]:
        logger.error(
            "场景 JSON 必须含 camera.intrinsics 和 camera.mount_pose,"
            f"当前 keys={list(scene['camera'].keys())}"
        )
        sys.exit(1)

    scene_K = scene["camera"]["intrinsics"]
    mount = scene["camera"]["mount_pose"]
    K_scene = {
        "fx": scene_K["fx"], "fy": scene_K["fy"],
        "cx": scene_K["cx"], "cy": scene_K["cy"],
    }
    T_scene = make_T(mount["xyz"], mount["rpy"])
    logger.info(
        f"[scene-K] fx={K_scene['fx']} fy={K_scene['fy']} "
        f"cx={K_scene['cx']} cy={K_scene['cy']} (source: {scene_K.get('source', '?')})"
    )
    logger.info(
        f"[scene-T] {mount['frame']} ← pose xyz={mount['xyz']} rpy={mount['rpy']}"
    )

    K_calibrated = cfg.HEAD_CAMERA_INTRINSICS
    T_calibrated = cfg.HEAD_TO_BASE_T
    logger.info(
        f"[calib-K] fx={K_calibrated['fx']} fy={K_calibrated['fy']} "
        f"cx={K_calibrated['cx']} cy={K_calibrated['cy']}"
    )

    # ── 2. 检测 ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("[detect] RGB → 螺母 + 收纳盒 像素")
    logger.info("=" * 70)
    nuts, box = detect_targets(rgb)
    logger.info(
        f"  找到 {len(nuts)} 个螺母 + 1 个箱子"
    )

    # ── 3. Mode A: scene intrinsics + calibrated T ──────────────
    logger.info("=" * 70)
    logger.info(
        "[Mode A] scene intrinsics (fx=fy=554.3) + calibrated HEAD_TO_BASE_T"
    )
    logger.info("=" * 70)
    res_A = evaluate_mode(
        "A", rgb, depth,
        intrinsics=K_scene,
        head_to_base_T=T_calibrated,
        nuts=nuts, box=box,
        scene_objects=scene["objects"],
    )
    log_mode("Mode A", res_A)

    # ── 4. Mode B: scene intrinsics + scene T(假设 obzg_33 == base_link)
    logger.info("=" * 70)
    logger.info(
        "[Mode B] scene intrinsics + scene camera mount pose "
        "(假设 mount link == base_link)"
    )
    logger.info("=" * 70)
    res_B = evaluate_mode(
        "B", rgb, depth,
        intrinsics=K_scene,
        head_to_base_T=T_scene,
        nuts=nuts, box=box,
        scene_objects=scene["objects"],
    )
    log_mode("Mode B", res_B)

    # ── 5. 旁注:calibrated K + calibrated T(原始)用于对照 ──────
    logger.info("=" * 70)
    logger.info(
        "[Mode R(ref)] calibrated K (fx=656 fy=862) + calibrated T "
        "—— 原始 calibrate 结果"
    )
    logger.info("=" * 70)
    res_R = evaluate_mode(
        "R", rgb, depth,
        intrinsics=K_calibrated,
        head_to_base_T=T_calibrated,
        nuts=nuts, box=box,
        scene_objects=scene["objects"],
    )
    log_mode("Mode R", res_R)

    # ── 6. 对比 ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("[对比] 同目标在三个模式下的 mm 误差")
    logger.info("=" * 70)
    header = f"{'target':>14s}  {'Mode A (scn K+cal T)':>22s}  {'Mode B (scn K+T)':>22s}  {'Mode R (cal K+T)':>22s}"
    logger.info(header)
    # 按 (kind, name) 索引
    idx_A = {(r["kind"], r["name"]): r for r in res_A}
    idx_B = {(r["kind"], r["name"]): r for r in res_B}
    idx_R = {(r["kind"], r["name"]): r for r in res_R}
    seen = list(idx_A.keys())
    for k in seen:
        a = idx_A.get(k, {}).get("error_mm")
        b = idx_B.get(k, {}).get("error_mm")
        r_ref = idx_R.get(k, {}).get("error_mm")
        a_s = f"{a:6.1f} mm" if a is not None else "    MISS"
        b_s = f"{b:6.1f} mm" if b is not None else "    MISS"
        r_s = f"{r_ref:6.1f} mm" if r_ref is not None else "    MISS"
        logger.info(f"{k[1]:>14s}  {a_s:>22s}  {b_s:>22s}  {r_s:>22s}")

    # ── 7. 推导 obzg_33 → base_link 实际变换 ─────────────────────
    # 由 calibrated T_cam_base 和 scene T_cam_obzg,反推 T_obzg_base
    # 公式:T_cam_base = T_obzg_base @ T_cam_obzg
    # → T_obzg_base = T_cam_base @ inv(T_cam_obzg)
    logger.info("=" * 70)
    logger.info(
        "[派生] 由 calibrated T_cam_base 和 scene T_cam_obzg 反推 obzg_33 → base_link"
    )
    logger.info("=" * 70)
    T_cam_base = cfg.HEAD_TO_BASE_T
    T_cam_obzg = T_scene
    T_obzg_base = T_cam_base @ np.linalg.inv(T_cam_obzg)

    pos = T_obzg_base[:3, 3]
    # 把旋转矩阵转成 RPY
    R = T_obzg_base[:3, :3]
    pitch = math.asin(-R[2, 0])
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    logger.info(f"  T_obzg_base:")
    logger.info(f"    position  = [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
    logger.info(f"    RPY(deg)  = [{math.degrees(roll):+.2f}, "
                 f"{math.degrees(pitch):+.2f}, {math.degrees(yaw):+.2f}]")
    logger.info(
        f"  → 如果 gazebo 把 obzg_33 当作 base_link 头部的固定 link,"
        f" 这就是它在 base_link 下的实际位姿。"
    )

    # ── 8. 用「scene K + 推导的 obzg_33→base + scene T_cam_obzg」= 重建的 T_cam_base
    # 实际上应该等于 calibrated T_cam_base,自检
    T_cam_base_rebuilt = T_obzg_base @ T_cam_obzg
    diff = np.linalg.norm(T_cam_base_rebuilt - T_cam_base)
    logger.info(
        f"  自检: T_obzg_base @ T_cam_obzg 与 calibrated T_cam_base 的 Frobenius 差 = {diff:.6f}"
        f" (理论上 = 0,因为我们就是从 calibrated T 反推的)"
    )

    # ── 9. 用 scene K + 重建的 T_cam_base 跑一次,验证等于 Mode A ───
    res_C = evaluate_mode(
        "C", rgb, depth,
        intrinsics=K_scene,
        head_to_base_T=T_cam_base_rebuilt,
        nuts=nuts, box=box,
        scene_objects=scene["objects"],
    )
    logger.info("=" * 70)
    logger.info(
        "[Mode C] scene intrinsics + 重建 T_cam_base "
        "(= T_obzg_base @ T_cam_obzg,自检跟 Mode A 等价)"
    )
    logger.info("=" * 70)
    log_mode("Mode C", res_C)
    diff_A_C = sum(
        abs(idx_A[k]["error_mm"] - res_C[i]["error_mm"])
        for i, k in enumerate(idx_A.keys())
        if idx_A[k]["error_mm"] is not None and res_C[i]["error_mm"] is not None
    )
    logger.info(f"  Mode A vs Mode C 总差 = {diff_A_C:.6f} mm (理论上 = 0)")


if __name__ == "__main__":
    main()