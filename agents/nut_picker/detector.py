"""OpenCV 规则识别:3 个螺母(按面积排大小)+ 1 个箱子(等分 3 个 cell)。

────────────────────────────────────────────────────────────────────────
为什么放 agents/nut_picker/ 而不是 drivers/?

HSV 阈值、面积阈值、箱子宽高比都是 nut_picker 私有的。其它 agent 不会复用。
等出现第二个 vision→base agent 再统一提到 drivers/。
────────────────────────────────────────────────────────────────────────

依赖:opencv-python-headless (已在 requirements.txt)。函数无 ROS 依赖,
可直接对 numpy 数组做检测,既能用 live frame 也能用已保存的 PNG。
"""

import logging

import cv2
import numpy as np

from . import config

logger = logging.getLogger("nut_picker.detector")


# ══════════════════════════════════════════════════════════════════════
#  自定义异常
# ══════════════════════════════════════════════════════════════════════
class DetectionError(RuntimeError):
    """检测未达预期(如少于 3 个螺母、找不到箱子)时抛出。"""


# ══════════════════════════════════════════════════════════════════════
#  螺母检测 (深灰/黑色螺母 + HoughCircles 找圆)
# ══════════════════════════════════════════════════════════════════════
def _nut_hsv_mask(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV → 按 NUT_HSV 取暗物体(深灰螺母 + 黑色框边)。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(
        hsv,
        np.array(config.NUT_HSV_LOW,  dtype=np.uint8),
        np.array(config.NUT_HSV_HIGH, dtype=np.uint8),
    )


def _box_hsv_mask(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV → 按 BOX_HSV 取浅蓝箱子。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.inRange(
        hsv,
        np.array(config.BOX_HSV_LOW,  dtype=np.uint8),
        np.array(config.BOX_HSV_HIGH, dtype=np.uint8),
    )


def _clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """OPEN 去噪点 + CLOSE 填小洞。kernel_size 越大,合并越厉害。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_nuts(rgb: np.ndarray) -> list[dict]:
    """先找黑色方框,再在框内(扣掉边框厚度)找螺母 → 按面积大/中/小。

    为什么先找框?
      螺母和框边都是深色,直接 findContours 会合并成 1 个 blob。
      框是规则的 4 顶点矩形,可以用 approxPolyDP 单独识别;识别后再在框内
      找螺母,可避免与框边混淆。

    返回:list[dict],长度恰好为 NUT_COUNT(默认 3),每个 dict 含:
        - label:       "big" | "medium" | "small"
        - centroid_px: (u, v)  浮点像素坐标
        - bbox:        (x, y, w, h)  像素
        - area_px:     int  轮廓面积

    异常:DetectionError — 找不到框或框内少于 NUT_COUNT 个螺母。
    """
    # 1. 找框:深色 mask 上最大的 4 顶点凸四边形
    mask = _clean_mask(_nut_hsv_mask(rgb), kernel_size=3)
    frame_bbox = _find_frame_bbox(mask)
    if frame_bbox is None:
        raise DetectionError(
            "螺母检测:找不到黑色方框(螺母容器)。检查 NUT_HSV_LOW/HIGH 或场景。"
        )
    fx, fy, fw, fh = frame_bbox
    pad = config.FRAME_INNER_PADDING
    ix1, iy1 = fx + pad, fy + pad
    ix2, iy2 = fx + fw - pad, fy + fh - pad
    if ix2 <= ix1 or iy2 <= iy1:
        raise DetectionError(
            f"螺母检测:框太小({fw}x{fh}),padding={pad} 后无可用内部区域。"
        )

    # 2. 框内找深色 blob
    inner_hsv = cv2.cvtColor(rgb[iy1:iy2, ix1:ix2], cv2.COLOR_RGB2HSV)
    inner_mask = cv2.inRange(
        inner_hsv,
        np.array(config.NUT_HSV_LOW,  dtype=np.uint8),
        np.array(config.NUT_HSV_HIGH, dtype=np.uint8),
    )

    contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 过滤掉面积太小 / 长宽比过大的(残留框边)
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.MIN_NUT_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / float(min(w, h))
        if aspect > config.NUT_MAX_ASPECT:
            continue
        kept.append((area, c, (x + ix1, y + iy1, w, h)))

    if len(kept) < config.NUT_COUNT:
        raise DetectionError(
            f"螺母检测:框内找到 {len(kept)} 个合格 blob(需要 {config.NUT_COUNT} 个)。"
            f"框 bbox=({fx},{fy},{fw},{fh}) inner=({ix1},{iy1},{ix2-ix1},{iy2-iy1})。"
            f"调 NUT_HSV_LOW/HIGH 或 MIN_NUT_AREA_PX 或 FRAME_INNER_PADDING。"
        )

    # 4. 按面积降序,前 NUT_COUNT 个 → big/medium/small
    kept.sort(key=lambda t: t[0], reverse=True)
    top = kept[:config.NUT_COUNT]
    labels = ["big", "medium", "small"]

    result = []
    for label, (area, cnt, bbox) in zip(labels, top):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        result.append({
            "label": label,
            "centroid_px": (cx, cy),
            "bbox": (x, y, w, h),
            "area_px": int(area),
            "contour": cnt,
        })
        logger.info(
            f"[detect] nut {label}: area={int(area)} centroid=({cx:.1f}, {cy:.1f}) "
            f"bbox=({x},{y},{w},{h})"
        )
    return result


def _find_frame_bbox(mask: np.ndarray):
    """在深色 mask 上找最大的 4 顶点凸四边形 bbox。返回 (x, y, w, h) 或 None。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_FRAME_AREA_PX:
            continue
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perim, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        bbox = cv2.boundingRect(approx)
        if best is None or area > best[0]:
            best = (area, bbox)
    return best[1] if best else None


# ══════════════════════════════════════════════════════════════════════
#  箱子检测(等分为 3 个 cell,基于浅蓝 HSV 颜色)
# ══════════════════════════════════════════════════════════════════════
def detect_box(rgb: np.ndarray) -> dict:
    """按浅蓝颜色找箱子,取最大凸四边形,等分为 3 个 cell(左→右)。

    返回 dict:
        - bbox:    (x, y, w, h)  像素
        - cells:   list[3]  每项 {centroid_px, bbox, index}
        - contour: np.ndarray  4 顶点矩形轮廓

    异常:DetectionError — 找不到合格的 4 顶点矩形。
    """
    mask = _clean_mask(_box_hsv_mask(rgb))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_BOX_AREA_PX:
            continue
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perim, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        aspect = w / float(h)
        if not (0.4 <= aspect <= 2.5):
            continue
        candidates.append((area, approx, (x, y, w, h)))

    if not candidates:
        raise DetectionError(
            "箱子检测:未找到 4 顶点矩形候选。检查 BOX_HSV_LOW/HIGH 或 MIN_BOX_AREA_PX。"
        )

    # 取面积最大的
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, contour4, (x, y, w, h) = candidates[0]

    # 等分为 3 个 cell(上→下):cell0 顶部 / cell1 中部 / cell2 底部
    cell_h = h // 3
    cells = []
    for i in range(3):
        cx_cell = x + w / 2.0
        cy_cell = y + i * cell_h + cell_h / 2.0
        cells.append({
            "index": i,
            "centroid_px": (cx_cell, cy_cell),
            "bbox": (x, y + i * cell_h, w, cell_h),
        })
        logger.info(f"[detect] box cell{i}: centroid=({cx_cell:.1f}, {cy_cell:.1f})")

    return {"bbox": (x, y, w, h), "cells": cells, "contour": contour4}


# ══════════════════════════════════════════════════════════════════════
#  可视化叠加(可选)
# ══════════════════════════════════════════════════════════════════════
def draw_overlay(rgb: np.ndarray, nuts: list[dict], box: dict) -> np.ndarray:
    """在 RGB 副本上画螺母 bbox/标签、箱子 bbox/cell 中心,返回 BGR 图像。"""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    colors = {"big": (0, 0, 255), "medium": (0, 165, 255), "small": (0, 255, 255)}
    for nut in nuts:
        x, y, w, h = nut["bbox"]
        c = colors[nut["label"]]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), c, 2)
        cv2.putText(
            bgr, f'{nut["label"]} area={nut["area_px"]}',
            (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
        )
    bx, by, bw, bh = box["bbox"]
    cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    for cell in box["cells"]:
        cx, cy = [int(v) for v in cell["centroid_px"]]
        cv2.circle(bgr, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(
            bgr, f'cell{cell["index"]}', (cx - 20, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return bgr


# ══════════════════════════════════════════════════════════════════════
#  调试可视化:即使检测失败也要能看出问题
# ══════════════════════════════════════════════════════════════════════
def draw_debug_overlay(
    rgb: np.ndarray,
    *,
    nut_mask: np.ndarray | None = None,
    box_mask: np.ndarray | None = None,
    raw_contours: list | None = None,
    kept_contours: list | None = None,
    box_candidates: list | None = None,
    nuts: list[dict] | None = None,
    box: dict | None = None,
    note: str = "",
) -> np.ndarray:
    """调试叠加图:三栏并排 [原图带标注 | nut_mask 灰度 | box_mask 灰度]。"""
    H, W = rgb.shape[:2]
    # 三栏横向拼接:原图 / nut_mask / box_mask
    panel = np.zeros((H, W * 3, 3), dtype=np.uint8)
    panel[:, :W, :] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if nut_mask is not None:
        m = cv2.resize(nut_mask, (W, H))
        panel[:, W:2*W, 0] = m
        panel[:, W:2*W, 1] = m
        panel[:, W:2*W, 2] = m
    cv2.putText(panel, "nut mask", (W + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if box_mask is not None:
        m = cv2.resize(box_mask, (W, H))
        panel[:, 2*W:, 0] = m
        panel[:, 2*W:, 1] = m
        panel[:, 2*W:, 2] = m
    cv2.putText(panel, "box mask", (2*W + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    left_view = panel[:, :W]   # 只在原图上画轮廓/bbox

    # 全部原始轮廓(红,细)
    if raw_contours:
        cv2.drawContours(left_view, raw_contours, -1, (0, 0, 200), 1)

    # 过面积阈值的轮廓(黄)
    if kept_contours:
        cv2.drawContours(left_view, kept_contours, -1, (0, 200, 200), 2)

    # box 候选(蓝)
    if box_candidates:
        for _, approx, _ in box_candidates:
            cv2.drawContours(left_view, [approx], -1, (200, 100, 0), 2)

    # 分类后的螺母(按大小上色 + 标签)
    if nuts:
        colors = {"big": (0, 0, 255), "medium": (0, 165, 255), "small": (0, 255, 255)}
        for n in nuts:
            x, y, w, h = n["bbox"]
            c = colors[n["label"]]
            cv2.rectangle(left_view, (x, y), (x + w, y + h), c, 2)
            cv2.putText(
                left_view, f'{n["label"]} a={n["area_px"]}',
                (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
            )

    # box
    if box:
        bx, by, bw, bh = box["bbox"]
        cv2.rectangle(left_view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        for cell in box["cells"]:
            cx, cy = [int(v) for v in cell["centroid_px"]]
            cv2.circle(left_view, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(
                left_view, f'cell{cell["index"]}', (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

    # 顶部提示
    if note:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            panel, note, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    # 图例(底部)
    legend_y = panel.shape[0] - 18
    legend_items = [
        ((0, 0, 200),   "raw contours"),
        ((0, 200, 200), "area-passed"),
        ((200, 100, 0), "box candidates"),
        ((0, 255, 0),   "box / cells"),
    ]
    x = 10
    for color, label in legend_items:
        cv2.rectangle(panel, (x, legend_y - 12), (x + 16, legend_y + 4), color, -1)
        cv2.putText(panel, label, (x + 22, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += 22 + len(label) * 9 + 12

    return panel


def inspect_nuts(rgb: np.ndarray):
    """调试辅助:返回 detect_nuts 过程中的中间产物,不 raise。

    返回 dict: {mask, raw_contours, kept_contours, nuts_or_none, error}
    """
    mask = _clean_mask(_nut_hsv_mask(rgb), kernel_size=3)
    raw_contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # HoughCircles 模式下没有"kept_contours"概念,返回空列表
    result = {"mask": mask, "raw_contours": raw_contours,
              "kept_contours": [], "nuts_or_none": None, "error": None}
    try:
        result["nuts_or_none"] = detect_nuts(rgb)
    except DetectionError as e:
        result["error"] = str(e)
    return result


def inspect_box(rgb: np.ndarray):
    """调试辅助:返回 detect_box 过程中的中间产物,不 raise。"""
    mask = _clean_mask(_box_hsv_mask(rgb))
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_BOX_AREA_PX:
            continue
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perim, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        aspect = w / float(h)
        if not (0.4 <= aspect <= 2.5):
            continue
        candidates.append((area, approx, (x, y, w, h)))
    result = {"mask": mask, "contours": contours,
              "box_candidates": candidates, "box_or_none": None, "error": None}
    try:
        result["box_or_none"] = detect_box(rgb)
    except DetectionError as e:
        result["error"] = str(e)
    return result