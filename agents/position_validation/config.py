"""position_validation 的全部常量与配置。

────────────────────────────────────────────────────────────────────────
职责边界:
  - 只放「position_validation 私有」的可调常量。
  - 相机 topics(同 nut_picker 的 head_*)、内参 K、head→base T 全部直接复用
    agents.nut_picker.config 的常量 — 不在这里重写一份。
  - 场景参数文件路径用一个环境变量 + 一个默认 fallback 控制。

scene_params.json 的格式(见 calibration_runs/position_validation_scene.json):

    {
      "camera": {
        "name": "head_camera",
        "position": [x, y, z],                # base_link 下 XYZ(米)
        "orientation_rpy": [roll, pitch, yaw] # 弧度
      },
      "objects": [
        {"name": "nut_big",   "kind": "nut", "xyz": [x, y, z]},
        {"name": "nut_medium","kind": "nut", "xyz": [x, y, z]},
        {"name": "nut_small", "kind": "nut", "xyz": [x, y, z]},
        {"name": "box",       "kind": "box", "xyz": [x, y, z]}
      ]
    }

运行时匹配策略:scene 里的每个对象按 `name` 或 `kind` 与检测结果对应 ——
  - kind=="nut" 顺序按 area 大→小对应 big/medium/small
  - kind=="box"  对应整箱中心(validation 只看 box 中心一个点)
────────────────────────────────────────────────────────────────────────
"""

import os

# 复用 nut_picker 的相机 topic(只用头部 RGB + 深度两路)
from agents.nut_picker.config import CAMERAS, WAIT_TIMEOUT

# 复用 nut_picker 的标定结果(K + head→base) — 这是当前唯一可用的标定
from agents.nut_picker.config import (
    HEAD_CAMERA_INTRINSICS,
    HEAD_TO_BASE_T,
    TABLE_Z_FALLBACK_M,
)


# ══════════════════════════════════════════════════════════════════════
#  场景参数 JSON 文件
# ══════════════════════════════════════════════════════════════════════
# POSITION_VALIDATION_SCENE_PARAMS 指向场景参数 JSON 路径;
# 留空 → 默认 calibration_runs/position_validation_scene.json
SCENE_PARAMS_PATH = os.getenv(
    "POSITION_VALIDATION_SCENE_PARAMS",
    "calibration_runs/position_validation_scene.json",
)


# ══════════════════════════════════════════════════════════════════════
#  可视化(可选)
# ══════════════════════════════════════════════════════════════════════
# 保存带标注的 overlay PNG 到 camera_frames/(0 = 不存)
SAVE_OVERLAY = os.getenv("POSITION_VALIDATION_SAVE_OVERLAY", "1") == "1"
OVERLAY_DIR = "camera_frames"
OVERLAY_PREFIX = "position_validation"


# ══════════════════════════════════════════════════════════════════════
#  误差阈值(用于日志分级的标签,不影响数据)
# ══════════════════════════════════════════════════════════════════════
# < EXCELLENT_MM 时打 [PASS]  (与真值几乎一致)
# < OK_MM         时打 [PASS]  (可接受)
# ≥ OK_MM         时打 [FAIL]  (需要复核标定/检测)
EXCELLENT_MM = 10.0
OK_MM = 50.0