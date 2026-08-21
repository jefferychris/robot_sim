# position_validation agent

把 head_rgb + head_depth 喂给现有的 `nut_picker.detector` + `geometry` 管线,
把每个螺母 / 收纳盒反投影到 base_link XYZ,再和「场景参数」里的真值逐个比对。

## 用途

回答一个问题:**当前标定(K + head→base T)+ 检测算法,解算出的 XYZ 到底离场景真值有多远?**

- 配合平台仿真:场景参数来自仿真场景 SDF/配置,可以拿到相机物理位姿 + 螺母/收纳盒的世界坐标
- 配合 raw depth 标定后的 nut_picker config:用来量化标定精度

## 运行

```bash
# 默认场景参数:calibration_runs/position_validation_scene.json
python main.py position_validation

# 自定义场景参数文件
POSITION_VALIDATION_SCENE_PARAMS=/path/to/scene.json python main.py position_validation

# 不保存 overlay PNG
POSITION_VALIDATION_SAVE_OVERLAY=0 python main.py position_validation
```

## 场景参数 JSON 格式

```json
{
  "camera": {
    "name": "head_rgb",
    "position": [x, y, z],
    "orientation_rpy": [roll, pitch, yaw]
  },
  "objects": [
    {"name": "nut_big",    "kind": "nut", "xyz": [x, y, z]},
    {"name": "nut_medium", "kind": "nut", "xyz": [x, y, z]},
    {"name": "nut_small",  "kind": "nut", "xyz": [x, y, z]},
    {"name": "box",        "kind": "box", "xyz": [x, y, z]}
  ]
}
```

- `camera.orientation_rpy` 单位:弧度
- `objects[*].kind`:`nut` / `box`
- 螺母会按 area 大→小对应 big/medium/small
- 收纳盒只比对箱子中心 1 个点

## 输出的状态等级

| error (mm) | 状态 | 颜色(overlay) |
|---|---|---|
| < 10  | `EXCELLENT` | 绿 |
| < 50  | `PASS` | 黄 |
| ≥ 50  | `FAIL` | 红 |
| 检测/反投影失败 | `MISS` | 灰 |

可在 `config.py` 调 `EXCELLENT_MM` / `OK_MM`。

## 输出示例

```
[scene] 加载 ...: camera=head_rgb, nuts=3, boxes=1
[position_validation] rgb.shape=(960, 540, 3)  depth.shape=(360, 640)
[EXCELLENT] nut nut_big        gt=(-0.229, -0.100, 0.282)  pred=(-0.224, -0.097, 0.283)  err=  6.2 mm
[   PASS ] nut nut_medium      gt=(-0.341, -0.171, 0.281)  pred=(-0.359, -0.173, 0.280)  err= 18.1 mm
[   FAIL ] nut nut_small       gt=(-0.298, -0.053, 0.287)  pred=(-0.231, -0.040, 0.290)  err= 67.5 mm
[EXCELLENT] box box            gt=(-0.310,  0.259, 0.300)  pred=(-0.305,  0.262, 0.302)  err=  5.9 mm
[summary] 4/4 有解; 平均=24.4mm 最大=67.5mm 最小=5.9mm; PASS=3 FAIL=1 MISS=0
```

overlay PNG 保存到 `camera_frames/position_validation_<ts>.png` —
RGB 上画每个对象的中心十字 + 误差数字 + 顶部摘要条。

## 文件

```
position_validation/
├── __init__.py              # run() 入口:相机 → 检测 → 比对 → 日志/overlay
├── config.py                # 复用 nut_picker 的 K/T/CAMERAS + 场景参数文件路径
├── validation.py            # load_scene_params / run_validation / summarize / log_results
├── validation_offline.py    # 离线版:读 saved RGB + raw depth 重算
├── validate_scene_intrinsics.py  # 双模式:scene K vs calibrated K,对照 scene GT
└── README.md                # 本文件
```

## 离线重算(`validation_offline.py`)

不需要 ROS/Gazebo 跑,直接读 `camera_frames/` 里 saved 文件复跑 pipeline:

```bash
# 默认读 camera_frames/head_rgb.png + head_depth_raw.npy
python -m agents.position_validation.validation_offline

# 打每个目标 RGB→depth→3D 的完整数据流
python -m agents.position_validation.validation_offline --verbose-flow

# 自定义路径
python -m agents.position_validation.validation_offline \
    --rgb-path /path/to/head_rgb.png \
    --depth-raw-path /path/to/head_depth_raw.npy \
    --scene-params /path/to/scene.json
```

**关键**:位置(3D XYZ)从 raw depth `.npy`(float32 米数)反投影得到,**不是** head_depth.png(归一化 0-255 已丢米数)。
RGB PNG 只用来定位「目标在哪一个像素」,**不参与** 3D 计算。

## 场景 intrinsics 验证(`validate_scene_intrinsics.py`)

回答:**「场景真值」推出来的 fx/fy 跟 calibrate.py 标定出来的 K,哪个误差更小?**

calibrate.py 算的 `HEAD_CAMERA_INTRINSICS={fx:656.84, fy:862.11}` 是非方像素;
按场景假设 `fx=fy=(W/2)/tan(h_fov/2)=554.3` 是方像素。两者都对不上,所以跑两组对照:

| Mode | K 来源 | T 来源 | 含义 |
|---|---|---|---|
| **A** | scene intrinsics (fx=fy=554.3) | calibrated HEAD_TO_BASE_T | 只换 K,看 intrinsic 误差占多少 |
| **B** | scene intrinsics | scene camera mount pose(假设 obzg_33==base_link) | 全用场景,看假设对不对 |
| **C** | scene intrinsics | 重建 T_cam_base = T_obzg_base @ T_cam_obzg | 推导 obzg_33→base_link 后重建 T,自检 ≡ Mode A |
| **R(ref)** | calibrated K | calibrated T | 现有 pipeline,作为 baseline |

```bash
python -m agents.position_validation.validate_scene_intrinsics
```

实测结果(基于 `camera_frames/` saved files + 当前 scene JSON):

```
        target    Mode A (scn K+cal T)        Mode B (scn K+T)        Mode R (cal K+T)
       nut_big                 96.9 mm                921.5 mm                 83.6 mm
    nut_medium                 44.5 mm               1104.2 mm                 27.1 mm
     nut_small                 69.2 mm               1019.9 mm                 58.1 mm
         box                112.2 mm                850.2 mm                 83.4 mm
```

结论:
- **Mode A vs Mode R**:scene K 平均 80.7mm,calibrated K 平均 63.0mm —— **calibrated K 反而更准 17mm**,
  说明 "fx=fy=554.3 (方像素假设)" 不是最优,calibrate.py 算的 `fx=657, fy=862` 才接近真实传感器
- **Mode B 误差 ~900mm**:`obzg_33` 不是 `base_link`(这是预期,见下方派生)
- **派生**:由 calibrated T 和 scene T 反推 `T_obzg_base = T_cam_base @ inv(T_cam_obzg)`
  ```
  T_obzg_base:
    position  = [-0.8349, -0.1037, +0.8945]
    RPY(deg)  = [-124.52, +32.23, -39.98]
  ```
  obzg_33 实际位于 base_link 左前方上方 89cm 处,带显著旋转
- **Mode A ≡ Mode C**(误差差 = 0 mm):确认 `T_cam_base = T_obzg_base @ T_cam_obzg` 自洽

## 场景参数 JSON 格式

```json
{
  "camera": {
    "name": "head_depth (dcam_obzg_33)",
    "intrinsics": {
      "fx": 554.3, "fy": 554.3, "cx": 320.0, "cy": 180.0,
      "resolution": [640, 360], "h_fov_rad": 1.047,
      "near_m": 0.1, "far_m": 10.0,
      "source": "derived from h_fov & W=640, H=360, square-pixel pinhole assumption"
    },
    "mount_pose": {
      "frame": "obzg_33",
      "xyz": [0.05, 0.0, -0.03],
      "rpy": [0.0, 0.8, 0.0],
      "source": "Gazebo scene, camera-to-mount-link local pose"
    }
  },
  "objects": [
    {"name": "nut_big",    "kind": "nut", "xyz": [-0.2286, -0.0999, 0.2815]},
    {"name": "nut_medium", "kind": "nut", "xyz": [-0.3413, -0.1710, 0.2806]},
    {"name": "nut_small",  "kind": "nut", "xyz": [-0.2975, -0.0527, 0.2868]},
    {"name": "box",        "kind": "box", "xyz": [-0.3104,  0.2586, 0.2996]}
  ]
}
```

兼容旧版格式: `camera.position` + `camera.orientation_rpy` 也接受(`__init__.run()` 用)。

字段说明:
- `camera.intrinsics`:来自场景(h_fov 推导,或平台 API 直接给)
- `camera.mount_pose`:相机在**挂载 link**(这里是 obzg_33)坐标系下的局部 pose
  - ⚠️ 不是 world/全局。要用到 base_link 需沿 `camera → mount_link → base_link → world` 链变换
- `camera.mount_pose.rpy`:rad
- `objects[*].kind`:`nut` / `box`
- 螺母按 area 大→小对应 big/medium/small
- 收纳盒只比对箱子中心 1 个点

## 输出的状态等级

| error (mm) | 状态 | 颜色(overlay) |
|---|---|---|
| < 10  | `EXCELLENT` | 绿 |
| < 50  | `PASS` | 黄 |
| ≥ 50  | `FAIL` | 红 |
| 检测/反投影失败 | `MISS` | 灰 |

可在 `config.py` 调 `EXCELLENT_MM` / `OK_MM`。