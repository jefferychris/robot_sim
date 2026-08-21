# 标定结果 — 给平台仿真用

## 1. config.py 要填的值

打开 `agents/nut_picker/config.py`,把下面两个常量替换为以下内容:

### `HEAD_CAMERA_INTRINSICS`

```python
HEAD_CAMERA_INTRINSICS = {
    "fx": 589.47,
    "fy": 407.37,
    "cx": 320.00,  # depth 宽度 640 的一半
    "cy": 180.00,  # depth 高度 360 的一半
}
```

### `HEAD_TO_BASE_T`

```python
HEAD_TO_BASE_T = np.array([
    [ 0.35569388, -0.82761364,  0.43420908, -0.54462108],
    [-0.93456917, -0.31104024,  0.17272645, -0.11249490],
    [-0.00789427, -0.46723617, -0.88409731,  0.91460053],
    [ 0.0,         0.0,         0.0,         1.0],
], dtype=np.float64)
```

## 2. 当前精度

| 点 | 误差 | 备注 |
|---|---|---|
| nut A | 87mm | 解算点 |
| nut B | 18mm | 解算点 |
| nut C | 67mm | 解算点 |
| cell0(箱顶) | 6mm | 跨域验证 |
| cell1(箱中) | 130mm | 箱 yaw=1.57 旋转,不配 box GT(用户忽略) |
| cell2(箱底) | 249mm | 同上 |

**螺母平均误差 ~57mm**(calibrate.py 自报 ~44mm,实际验证含逐点比较),够用来验证仿真流程。

## 3. ⚠️ 本次标定的精度边界(实测)

**用了改进后的 calibrate.py 重跑,精度卡在 ~44mm 上不去,根本原因是 PNG 归一化**:

- 算法改动都做了:邻域中位数(降噪) + RANSAC(去 outlier)
- 算法改动都无效:cx/cy 加入搜索(过拟合,主点跑到 416/126) + 自动估 near/far(退化到 K 边界)
- 真正的瓶颈:PNG 深度是 0-255 归一化的,真实米数已丢,反推 [0.55, 0.85] 区间外的信息不可逆损失
- 4 点 GT(3 螺母 + 1 箱)在丢失了米数范围的深度上,无法同时确定 fx/fy 和近/远米数

**真要精度 < 20mm,只能在平台侧跑一次,得到 raw .npy**:

1. `python main.py camera_demo`  ← 现在默认就存 `head_depth_raw.npy`(无需环境变量)
2. 用现有 `gt.json` 重跑 calibrate.py,自动用 raw 路径

## 4. 平台侧执行清单

```bash
# A. 拿到 raw depth(默认就存,不用设环境变量)
python main.py camera_demo
# 产物: camera_frames/head_rgb.png + head_depth.png + head_depth_raw.npy ★
```

```bash
# B. 用 raw depth 重做高精度标定
python -m agents.nut_picker.calibrate \
    --gt-json calibration_runs/gt.json \
    --depth-raw-path camera_frames/head_depth_raw.npy \
    --rgb-coord-system raw \
    --print-config
# 把新 K/T 贴回 config.py
```

```bash
# C. sim 模式验证(不实际抓,只验证可达性)
python main.py
# 看日志,pose_check 是否还报 out_of_workspace
# ⚠️ raw depth 标定后 nut Z ≈ 0.28m,APPROACH_HEIGHT_M 已从 0.15 降到 0.05,
# approach Z = 0.33m 落在 LinkerArmA7 工作空间内(0.15 时是 0.43m 会超)
```

```bash
# D. dry-run 验证(检测+打印动作序列,不动作)
NUT_PICKER_DRY_RUN=1 python main.py
# 期望:plan 输出 xyz 与 GT 偏差 < 5cm
```

```bash
# E. sim 全流程(实际抓取)
python main.py
# 期望:3 个螺母按 big→cell0, medium→cell1, small→cell2 顺序被放入
```

## 5. 如果还报错

| 现象 | 可能原因 | 下一步 |
|---|---|---|
| pose_check 仍 out_of_workspace | 标定有偏差,目标点还在臂展外 | 用 raw depth 重标定(精度差 ~5mm vs PNG ~44mm) |
| 抓偏 / 放偏 | APPROACH/GRASP/PLACE 高度需微调 | 改 `APPROACH_HEIGHT_M=0.05`, `GRASP_HEIGHT_M=0.005`, `PLACE_HEIGHT_M=0.01` |
| 手抓空(grasp_force 返回 False) | 抓取力太小或物体太小 | 调 `GRASP_FORCE=80` |
| detection 失败(DetectionError) | HSV 阈值不对(场景光照变了) | 跑 `python -m agents.nut_picker.debug_detect` 看调试图 |

## 文件清单

```
calibration_runs/
├── gt.json          # 输入:4 个 GT 点
├── result.txt       # 完整 calibrate.py 输出(含配置建议)
└── README.md        # 本文件
```