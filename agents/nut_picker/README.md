# nut_picker agent

用头部相机识别 3 个螺母(大/中/小)+ 一个 3 格箱子,用左臂 LinkerArmA7 + 左手 LinkerHandO6Left 把螺母按大小顺序放进对应格子。

## 运行

```bash
# 默认走 main.py(DEFAULT_AGENT 已改为 "nut_picker")
python main.py

# 只跑检测+打印动作序列,不实际驱动机械臂
NUT_PICKER_DRY_RUN=1 python main.py
```

## 标定 TODO(首次跑前必看)

下面几个量在 `config.py` 顶部标了 TODO,直接影响能不能抓准。第一次跑之前请先确认:

### 1. 相机内参(`HEAD_CAMERA_INTRINSICS`)

当前值(标定得出,占位 → 见下方的「标定流程」重新求解):

```python
HEAD_CAMERA_INTRINSICS = {
    "fx": 589.47,
    "fy": 407.37,
    "cx": 320.00,  # depth 宽度 640 的一半(主点近似中心)
    "cy": 180.00,  # depth 高度 360 的一半
}
```

也可直接从 `/camera_info` 取:

```bash
ros2 topic list | grep camera_info
ros2 topic echo <topic> --once
```

拿到 `K[0,0]/K[1,1]/K[0,2]/K[1,2]` 后填回 `config.py`。

### 2. head→base 4×4 变换(`HEAD_TO_BASE_T`)

当前值(标定得出)。在 nut 位置误差最大 87mm,平均 57mm。

### 标定流程(新场景/换相机时重做)

#### 步骤 1:平台侧保存 raw depth
```bash
SAVE_DEPTH_NPY=1 python main.py camera_demo
```
产物:
- `camera_frames/head_rgb.png`(1920×1080,或 saved 后 960×540)
- `camera_frames/head_depth_raw.npy`(原始 float32,米数)

#### 步骤 2:从平台 API 拿 ground truth
至少 4 个点(3 螺母 + 1 箱),每个含 `name` + `xyz` + `rgb_uv`。`rgb_uv` 是
原始 RGB 像素坐标(1920×1080),脚本会自动缩到 saved 图坐标系。

```json
[
  {"name": "nutA", "xyz": [-0.2286, -0.0999, 0.2815], "rgb_uv": [1087, 700]},
  {"name": "nutB", "xyz": [-0.3413, -0.1710, 0.2806], "rgb_uv": [1340, 793]},
  {"name": "nutC", "xyz": [-0.2975, -0.0527, 0.2868], "rgb_uv": [1168, 581]},
  {"name": "box",  "xyz": [-0.3104,  0.2586, 0.2996], "rgb_uv": [331, 505]}
]
```

#### 步骤 3:跑标定
```bash
python -m agents.nut_picker.calibrate \
    --gt-json gt.json \
    --depth-raw-path camera_frames/head_depth_raw.npy \
    --rgb-coord-system raw \
    --print-config
```

输出 `HEAD_CAMERA_INTRINSICS` + `HEAD_TO_BASE_T`,贴回 `config.py`。

**注意**:此标定**不能**用 saved PNG depth(已归一化丢失米数),必须用 raw .npy。
若没有 raw .npy,可加 `--depth-near-m 0.55 --depth-far-m 0.85` 临时凑合用(精度差)。

### 3. 螺母识别(`NUT_HSV_*` + `FRAME_INNER_PADDING`)

当前场景:**螺母是深灰色(HSV ≈ (0, 0, 59))**,放在一个**黑色方框**里。

策略:**先找黑色方框(最大 4 顶点凸四边形),再在框内找螺母**。
直接对全图找轮廓会把螺母和框边合并成一个 blob,无法分离。

```python
NUT_HSV_LOW  = (0,   0,   0)
NUT_HSV_HIGH = (179, 255, 120)   # V 上限:排除桌子/箱体亮区
FRAME_INNER_PADDING = 35         # 框内缩像素,需 >= 框边厚度
MIN_NUT_AREA_PX = 500
NUT_MAX_ASPECT = 3.0             # 过滤细长残留框边
```

调整:
- **框边厚度变了** → 改 `FRAME_INNER_PADDING`(场景里 ~30-40px)
- **螺母颜色变了** → 改 `NUT_HSV_HIGH`(值越小匹配越暗)
- **小螺母被滤掉** → 改 `MIN_NUT_AREA_PX`

### 4. 箱子 HSV 范围(`BOX_HSV_LOW`/`HIGH`)

当前场景:浅蓝色塑料箱(HSV ≈ (109, 214, 255))。

```python
BOX_HSV_LOW  = (100,  30, 100)
BOX_HSV_HIGH = (130, 255, 255)
```

调整:H 范围(±10)、S/V 上下限(按实物)。

### 5. 运动高度(`APPROACH_HEIGHT_M` / `GRASP_HEIGHT_M` / `PLACE_HEIGHT_M`)

首次 sim 全流程后:

```python
# 在 pick 成功后 arm.get_pose() 看一下实际 Z
pose, _ = arm.get_pose()
print(pose[2])  # 当前 Z
```

按需调整三个常量。

## 检测调参

`detector.py` 用 OpenCV 规则识别,所有阈值都在 `config.py`:

| 常量 | 作用 |
| --- | --- |
| `NUT_HSV_LOW/HIGH` | 螺母+框边的深色范围 |
| `FRAME_INNER_PADDING` | 框内缩像素(>= 框边厚度) |
| `MIN_NUT_AREA_PX` | 框内深色 blob 最小面积 |
| `NUT_MAX_ASPECT` | 螺母 bbox 长宽比上限 |
| `BOX_HSV_LOW/HIGH` | 箱子颜色范围 |
| `MIN_BOX_AREA_PX` | 箱子轮廓最小面积 |
| `MIN_FRAME_AREA_PX` | 黑色框最小面积 |

### 离线调试图(不需 ROS,直接看检测效果)

```bash
# 默认读 camera_frames/head_rgb.png,输出到 camera_frames/nut_picker_debug_<ts>.png
python -m agents.nut_picker.debug_detect

# 指定输入图 / 输出路径 / 备注
python -m agents.nut_picker.debug_detect -i some.png -o out.png -n "after HSV tune"
```

调试图是**三栏横向拼接**:左原图带各种标注 / 中 nut mask 灰度 / 右 box mask 灰度。
固定图例(底部):

- 红色细框:`findContours` 全部原始轮廓
- 黄色粗框:过面积阈值的轮廓
- 蓝色粗框:`detect_box` 内部的 4 顶点矩形候选
- 绿色:最终 box / cell
- 红/橙/黄大框:分类后的 big / medium / small 螺母

### 自动调试图(`run()` 内置)

无论检测成败都会保存:

- 成功 → `camera_frames/nut_picker_overlay_<ts>.png`(只画最终结果)
- 失败 → `camera_frames/nut_picker_fail_<ts>.png`(调试图 + 顶部黑色失败原因)

环境变量:

- `NUT_PICKER_SAVE_OVERLAY=0` 关掉成功的覆盖图(失败的调试图仍会保存)
- `NUT_PICKER_DRY_RUN=1` 只检测+打印动作序列,不实际驱动机械臂

## 流程

```
head_rgb + head_depth
  ↓ detector.detect_nuts (深色 mask → 找黑色框 → 框内找 blob → 面积排序大/中/小)
  ↓ detector.detect_box (浅蓝 HSV mask → 最大 4 顶点凸四边形 → 等分 3 cell)
  ↓ geometry.pixel_to_base (RGB 像素 → depth → 相机内参 → head→base)
  ↓ motion.PickPlaceRunner (pick big → place cell0 → pick medium → place cell1 → pick small → place cell2)
```

## 故障排查

- **"螺母检测:框内找到 N 个合格 blob"**:FRAME_INNER_PADDING 太小把框边当成螺母/太大框被吞掉;
  或 MIN_NUT_AREA_PX 太大把小螺母滤掉。看 overlay PNG 调 `config.py`
- **"找不到黑色方框"**:场景里没方框,或 NUT_HSV_HIGH 太严把框边 V=41 也排除了。改成 V>=120
- **"箱子检测:未找到 4 顶点矩形候选"**:箱子不是浅蓝色,改 BOX_HSV_LOW/HIGH
- **pose_check 不可达**:目标点超出 LinkerArmA7 工作空间,通常说明内参或 head→base 没标定好
- **grasp_force 返回 False**:GRASP_FORCE 太小或目标尺寸超出夹爪,加大 strength 或重新识别
- **move_to 一直抛 IKSolutionError**:目标点奇异,先 `arm.home()` 重置姿态
- **arm_hand_demo 是坏的**:它调 `hand.clench()` 会 AttributeError,**别用它的运动代码**,只用 `LEFT_ARM_ID` / `LEFT_HAND_ID` 常量

## 文件

```
nut_picker/
├── __init__.py    # run() 编排入口
├── config.py      # 全部可调常量 + 标定后的 K / head→base
├── detector.py    # OpenCV 规则识别(nuts/box)
├── geometry.py    # 像素→base 坐标变换
├── motion.py      # PickPlaceRunner:pick/place/home + 重试
├── calibrate.py   # 用 GT + raw depth 标定 K + head→base 的脚本
├── debug_detect.py # 离线调试图(无需 ROS)
└── README.md      # 本文件
```