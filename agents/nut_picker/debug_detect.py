"""离线调试脚本:对 camera_frames/head_rgb.png 跑检测,无论成败都保存调试图。

不需要 ROS、不需要机械臂。直接:
    python -m agents.nut_picker.debug_detect

默认输入:camera_frames/head_rgb.png
默认输出:camera_frames/nut_picker_debug_<ts>.png(全屏:原图+mask+候选+结果)

可选参数:
    --input PATH    指定输入 RGB 图(默认 camera_frames/head_rgb.png)
    --output PATH   指定输出图路径(默认 camera_frames/nut_picker_debug_<ts>.png)
    --note "..."    写到图顶部的备注文字
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

# 允许以模块方式运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.nut_picker import config, detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(config.OVERLAY_DIR, "head_rgb.png"),
        help="输入 RGB 图路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出调试图路径(默认 camera_frames/nut_picker_debug_<ts>.png)",
    )
    parser.add_argument(
        "--note", "-n",
        default="",
        help="写到图顶部的备注文字",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[debug_detect] 找不到输入图: {args.input}", file=sys.stderr)
        sys.exit(1)

    rgb = np.array(Image.open(args.input).convert("RGB"), dtype=np.uint8)
    print(f"[debug_detect] 加载: {args.input}  shape={rgb.shape}")

    n_info = detector.inspect_nuts(rgb)
    b_info = detector.inspect_box(rgb)

    print(f"[debug_detect] HSV mask 非零像素: {n_info['mask'].sum()}/{n_info['mask'].size} "
          f"({100*n_info['mask'].sum()/max(1,n_info['mask'].size):.2f}%)")
    print(f"[debug_detect] 原始轮廓: {len(n_info['raw_contours'])}, "
          f"过面积阈值: {len(n_info['kept_contours'])}, "
          f"箱候选: {len(b_info['box_candidates'])}")

    note = args.note or ""
    if n_info["error"]:
        note = (note + " | " if note else "") + f"nuts: {n_info['error']}"
    if b_info["error"]:
        note = (note + " | " if note else "") + f"box: {b_info['error']}"
    if not n_info["error"] and not b_info["error"]:
        if not note:
            note = "DETECTION OK"

    panel = detector.draw_debug_overlay(
        rgb,
        nut_mask=n_info["mask"],
        box_mask=b_info["mask"],
        raw_contours=n_info["raw_contours"],
        kept_contours=n_info["kept_contours"],
        box_candidates=b_info["box_candidates"],
        nuts=n_info["nuts_or_none"],
        box=b_info["box_or_none"],
        note=note,
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = args.output or os.path.join(config.OVERLAY_DIR, f"nut_picker_debug_{ts}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, panel)
    print(f"[debug_detect] 调试图已保存: {out}")
    print(f"[debug_detect] {note}")


if __name__ == "__main__":
    main()