"""相机数据获取演示: 订阅头部与左右手末端 RGB/深度相机, 打印并保存最新帧。"""

import os
import time

import numpy as np

from drivers.camera import CameraHub, save_frame

from . import config


def run():
    hub = CameraHub(config.CAMERAS, node_name="camera_demo")
    print(f"[camera_demo] 已订阅 {len(config.CAMERAS)} 路相机:")
    for name, topic in config.CAMERAS.items():
        print(f"  - {name}: {topic}")

    # 等待所有相机至少各收到一帧(最多 WAIT_TIMEOUT 秒)
    deadline = time.time() + config.WAIT_TIMEOUT
    while time.time() < deadline:
        frames = hub.get_all()
        missing = [n for n in config.CAMERAS if n not in frames]
        if not missing:
            break
        time.sleep(0.2)

    frames = hub.get_all()
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    save_raw_depth = os.getenv("SAVE_DEPTH_NPY", "0") == "1"
    print("=" * 70)
    for name in config.CAMERAS:
        frame = frames.get(name)
        if frame is None:
            print(f"[{name}] ✗ 未收到数据 (topic={config.CAMERAS[name]})")
            continue
        arr = frame["array"]
        print(f"[{name}] topic={config.CAMERAS[name]}")
        print(f"  encoding={frame['encoding']}  shape={arr.shape}  dtype={arr.dtype}")
        print(f"  frame_id={frame['frame_id']}  stamp={frame['stamp']:.3f}")
        if arr.dtype == np.float32 and arr.ndim == 2:
            valid = arr[np.isfinite(arr)]
            if valid.size:
                print(
                    f"  深度 min={valid.min():.3f}m max={valid.max():.3f}m "
                    f"mean={valid.mean():.3f}m "
                    f"(有效像素 {valid.size}/{arr.size})"
                )
            else:
                print("  深度 无有效测量(全部 inf/nan)")
        elif arr.ndim == 3:
            print(
                f"  像素均值 R={arr[..., 0].mean():.1f} "
                f"G={arr[..., 1].mean():.1f} B={arr[..., 2].mean():.1f}"
            )
        path = os.path.join(config.SAVE_DIR, f"{name}.png")
        try:
            save_frame(frame, path)
            print(f"  已保存: {path}")
        except Exception as e:
            print(f"  保存失败: {e}")
        # 保存原始深度为 .npy(标定用),默认关闭
        if save_raw_depth and arr.dtype == np.float32 and arr.ndim == 2:
            npy_path = os.path.join(config.SAVE_DIR, f"{name}_raw.npy")
            np.save(npy_path, arr)
            print(f"  原始深度已保存: {npy_path}")
    print("=" * 70)

    result = hub.get_all()   # 最新帧快照, 供编程复用
    hub.shutdown()
    return result
