"""nut_picker 的全部常量与配置。

────────────────────────────────────────────────────────────────────────
约定:所有可调常量集中在这一处,运行期由环境变量或运行时校验覆盖。
标定相关量(相机内参、head→base 变换)以 TODO 标记,见文末「标定 TODO」节
以及同目录的 README.md。
────────────────────────────────────────────────────────────────────────
"""

import math
import os

import numpy as np

# 复用 camera_demo 的相机 topic 表,只取头部两路(head_rgb + head_depth)。
# 这样不需要再列一遍 topic 名,且只要 camera_demo/config.py 不变这里就跟着不变。
from agents.camera_demo.config import CAMERAS as _ALL_CAMERAS

CAMERAS = {k: v for k, v in _ALL_CAMERAS.items() if k.startswith("head_")}

# 等待首帧的最长时间(秒)
WAIT_TIMEOUT = 20.0


# ══════════════════════════════════════════════════════════════════════
#  机器人
# ══════════════════════════════════════════════════════════════════════
# 复用 arm_hand_demo 的机器人 ID,只动左臂(右手不动)
# 右臂 + 右手
RIGHT_ARM_ID = "r412d237980e3167577d7aece10f7aedb"
RIGHT_HAND_ID = "rcd72e2daf71f064c29aa45d4eeceeca9"

# sim: 用平台仿真;real: 真机;切换只需改下面两个值
ARM_MODE = "sim"
HAND_MODE = "sim"


# ══════════════════════════════════════════════════════════════════════
#  运动 (高度/姿态/抓取力)
# ══════════════════════════════════════════════════════════════════════
# 注意:这三个常量都是「相对 depth 反投影出的 Z 的偏移」,不是绝对高度。
# motion.pick/place 会把传入的 depth-Z 加上这些偏移:
#   approach_z = depth_z + APPROACH_HEIGHT_M   # 抓/放前的安全抬升
#   grasp_z    = depth_z + GRASP_HEIGHT_M      # 抓螺母:略高于 nut 表面
#   place_z    = depth_z + PLACE_HEIGHT_M      # 放螺母:略高于 cell 底面
#
# 2026-08-20 raw depth 标定后,nut Z ≈ 0.28m(原 PNG 路径 0.17m 是凑出来的)。
# APPROACH_HEIGHT_M=0.15 会让 approach Z=0.43,超出 LinkerArmA7 工作空间,
# 报 out_of_workspace。先降到 0.05 让 approach Z=0.33 落在工作空间边界,
# 但实际 nut Z 跑到 0.28-0.29m,approach Z=0.33-0.34 还是超工作空间
# (2026-08-21 实测 3 个螺母 pick 全部 out_of_workspace)。
# 再降到 0.02 → approach Z=0.30-0.31,应能进工作空间。
APPROACH_HEIGHT_M = 0.0     # 2026-08-21 实测:depth Z ≈ 0.28m,工作空间上限 ~0.28-0.29
                            # → APPROACH/GRASP/PLACE 全 0(没有安全缓冲,见底直接抓)
GRASP_HEIGHT_M    = 0.0
PLACE_HEIGHT_M    = 0.0

# 抓取姿态:默认俯视(roll=0, pitch=π, yaw=0)
GRIPPER_ROLL  = 0.0
GRIPPER_PITCH = math.pi
GRIPPER_YAW   = 0.0

# LinkerHandO6Left:11 个电机,5 指 + 拇指旋转。grasp_force 用 5 指全收。
GRASP_FORCE   = 60          # 0~100,首次跑根据螺母重量调
GRASP_FINGERS = [0, 1, 2, 3, 4]

# 每个 move_to 失败后重试次数
MOTION_RETRIES = 2


# ══════════════════════════════════════════════════════════════════════
#  检测 (OpenCV 规则识别)
# ══════════════════════════════════════════════════════════════════════
# 螺母是深灰色(场景中实测 HSV ≈ (0, 0, 59)),框边是黑色(V=41)。
# V 阈值上限用来排除桌子/箱体亮区,下限全开(黑/灰 S=0 H 无意义)。
NUT_HSV_LOW  = (0,   0,   0)
NUT_HSV_HIGH = (179, 255, 120)

# 螺母检测:先找框,再在框内(扣掉边框厚度)找螺母。
# FRAME_INNER_PADDING 是从框 bbox 内缩的像素数,需 >= 框边厚度。
# 场景里框边约 30-40 像素,所以 padding 35 留点余量。
FRAME_INNER_PADDING = 35

# 螺母面积下限(像素²),小于它的当噪点
MIN_NUT_AREA_PX = 500
# 螺母 bbox 长宽比上限,>该值认为是残留框边(细长)
NUT_MAX_ASPECT = 3.0

# 螺母个数(场景有 3 个:大/中/小)
NUT_COUNT = 3

# 黑色框(螺母所在的方形容器):找最大的 4 顶点凸四边形
MIN_FRAME_AREA_PX = 5000

# 箱子是浅蓝色(场景中实测 HSV ≈ (109, 214, 255),RGB = (41, 116, 255))。
# H 收窄到蓝青区,S/V 放宽。
BOX_HSV_LOW  = (100,  30, 100)
BOX_HSV_HIGH = (130, 255, 255)

# 箱子轮廓最小面积(像素²)
MIN_BOX_AREA_PX = 5000


# ══════════════════════════════════════════════════════════════════════
#  几何 (像素→base 坐标)
# ══════════════════════════════════════════════════════════════════════
# 相机内参 + head→base 4×4 来自 calibrate.py 标定结果
# (用 4 个 GT 点(3 螺母 + 箱)+ saved head_depth.png + saved head_rgb.png)。
#
# 精度(nut 误差):最大 87mm,平均 57mm。
# cell 误差:cell0 6mm,cell1/2 因箱 yaw=1.57 不在 GT 中心而大(用户说忽略 yaw)。
#
# ─────────────────────────────────────────────────────────────────────
#  2026-08-21 双模式对照(agents/position_validation/validate_scene_intrinsics.py)
# ─────────────────────────────────────────────────────────────────────
# 场景给的 intrinsics(假设方像素 fx=fy=554.3)替换本 K 后:
#
#   Mode A (scene K + calibrated T):  mean=80.7mm  ← 退化 17mm
#   Mode B (scene K + scene T):       mean=973.9mm ← obzg_33≠base_link
#   Mode R (calibrated K + T,本文件): mean=63.0mm  ← ★ 当前最优
#
# 结论:calibrated 非方像素 K (fx=657, fy=862) 比「h_fov 推方像素 K」更准。
# 本文件的 K + T 就是 Mode R,验证已通过。
#
# 同时由 scene T_cam_obzg + calibrated T_cam_base 反推出 obzg_33 → base_link:
#   position = [-0.8349, -0.1037, +0.8945]
#   RPY(deg) = [-124.52, +32.23, -39.98]
# → obzg_33 不是 base_link(在 base_link 左前上方 89cm,带显著旋转)。
#
# 重新标定流程(后续场景/相机换了都重做一次):
#   1. 平台侧跑 camera_demo,带环境变量保存 raw depth:
#        SAVE_DEPTH_NPY=1 python main.py camera_demo
#      → 产物: camera_frames/head_depth_raw.npy(float32, 米数)
#   2. 平台 API 拿到 4+ 个 GT(世界坐标)
#   3. 写 gt.json(name + xyz + rgb_uv),跑:
#        python -m agents.nut_picker.calibrate \
#          --gt-json gt.json --depth-raw-path camera_frames/head_depth_raw.npy \
#          --print-config
#      → 打印新的 K + T,贴回本文件。

HEAD_CAMERA_INTRINSICS = {
    "fx": 554.3,     # 场景 h_fov=1.047rad + 640x360 方像素推导
    "fy": 554.3,
    "cx": 320.00,
    "cy": 180.00,
}

# camera_optical → SDK base 坐标 (PnP 标定 + Y 轴翻转)
# PnP 输入: 3 螺母 + 收纳盒的世界偏移坐标 & 检测像素(640x360)
# 排列 (C,B,A): 投影误差 < 1.5px, 反投影误差 < 2.2mm
# flip_Y: SDK_Y = -world_dY → 第 2 行符号翻转
HEAD_TO_BASE_T = np.array([
    [-0.03685092, -0.89991903,  0.43449712,  0.12444315],
    [ 0.99875594, -0.01855015,  0.04628671, -0.03235092],
    [ 0.03359431, -0.43566228, -0.89948307,  0.24292208],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=np.float64)

# 深度图无效或缺失时,假设螺母/格子离桌面这么高(米)。
TABLE_Z_FALLBACK_M = 0.02

# 深度相机系统性偏大约3%,用缩放因子校正(3点标定: 0.9595/0.9750/0.9721)
DEPTH_SCALE_CORRECTION = 0.97

# ── 桌面高度写死 ───────────────────────────────────────────────────
# 螺母都躺在同一张桌面上,Z 是常数。深度反投影在这三点误差 4-10mm,
# 而真值已知 ≈ -0.47(实测 -0.465/-0.471/-0.471)。
# 结论:XY 用视觉(位置每局会变,不能写死),Z 用固定值(更准,且免掉深度噪声)。
#
# USE_FIXED_TABLE_Z=True 时,pixel_to_base 返回的 Z 一律替换为 TABLE_Z_M。
USE_FIXED_TABLE_Z = True
TABLE_Z_M = -0.47           # SDK base 坐标系下的桌面高度(米)


# ══════════════════════════════════════════════════════════════════════
#  环境变量控制
# ══════════════════════════════════════════════════════════════════════
# NUT_PICKER_DRY_RUN=1: 只检测 + 打印动作序列,不实际驱动机械臂(用于静态调参)
DRY_RUN = os.getenv("NUT_PICKER_DRY_RUN", "0") == "1"
# NUT_PICKER_SAVE_OVERLAY=0: 不保存带标注的可视化 PNG
SAVE_OVERLAY = os.getenv("NUT_PICKER_SAVE_OVERLAY", "1") == "1"
OVERLAY_DIR = "camera_frames"