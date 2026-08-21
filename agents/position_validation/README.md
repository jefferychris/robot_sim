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
├── __init__.py          # run() 入口:相机 → 检测 → 比对 → 日志/overlay
├── config.py            # 复用 nut_picker 的 K/T/CAMERAS + 场景参数文件路径
├── validation.py        # load_scene_params / run_validation / summarize / log_results
├── validation_offline.py # 离线版:读 saved RGB + raw depth 重算(显式区分 RGB/depth)
└── README.md            # 本文件
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
RGB PNG 只用来定位「目标在哪一个像素」,**不参与** 3D 计算。日志会显式标出:

```
[load] RGB ← camera_frames/head_rgb.png  shape=(540, 960, 3) dtype=uint8(仅用于找目标像素)
[load] raw depth ← camera_frames/head_depth_raw.npy  shape=(360, 640) dtype=float32(用于 3D 反投影)
...
  → RGB 只用来找目标像素,不参与 3D 计算
  → raw depth 提供 z 值(米)→ 针孔反投影 → head→base T → base_link XYZ
...
# --verbose-flow 多打一行:
  nut_big: rgb_px=(544,350)  depth_px=(362.3,233.3)  depth=0.7855m  base=(-0.305,-0.067,0.284)
```