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

**螺母平均误差 ~57mm**,够用来验证仿真流程。

## 3. ⚠️ 本次标定的局限

- **用的是 saved PNG depth(已归一化 0-255)**,精度受限
- **真标定需要在平台侧重做**:
  1. `SAVE_DEPTH_NPY=1 python main.py camera_demo` 拿到 raw depth .npy
  2. 重跑 calibrate.py,误差有望降到 <20mm

## 4. 平台侧执行清单

```bash
# A. sim 模式验证(不实际抓,只验证可达性)
python main.py
# 看日志,pose_check 是否还报 out_of_workspace
# 期望:之前 out_of_workspace 消失 → 都可达

# B. dry-run 验证(检测+打印动作序列,不动作)
NUT_PICKER_DRY_RUN=1 python main.py
# 期望:plan 输出 xyz 与本次 ground truth 偏差 < 10cm(取整到 mm)

# C. sim 全流程(实际抓取)
python main.py
# 期望:3 个螺母按 big→cell0, medium→cell1, small→cell2 顺序被放入
```

## 5. 如果还报错

| 现象 | 可能原因 | 下一步 |
|---|---|---|
| pose_check 仍 out_of_workspace | 标定有偏差,目标点还在臂展外 | 用 raw depth 重标定 |
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