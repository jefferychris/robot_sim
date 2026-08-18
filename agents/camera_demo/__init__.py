"""相机数据获取演示: 订阅头部与左右手末端 RGB/深度相机, 打印最新帧信息。"""

import time

import numpy as np

from drivers.camera import CameraHub

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
            print(
                f"  深度 min={arr.min():.3f}m max={arr.max():.3f}m "
                f"mean={arr.mean():.3f}m"
            )
        elif arr.ndim == 3:
            print(
                f"  像素均值 R={arr[..., 0].mean():.1f} "
                f"G={arr[..., 1].mean():.1f} B={arr[..., 2].mean():.1f}"
            )
    print("=" * 70)

    result = hub.get_all()   # 最新帧快照, 供编程复用
    hub.shutdown()
    return result
