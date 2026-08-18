"""可复用的相机(图像/深度)订阅驱动。

订阅一组 ROS2 图像 topic, 后台线程收帧, 把每路"最新帧"缓存到内存,
供任意 agent 通过 get_frame() / get_all() 取用。

消息类型: sensor_msgs/Image
  - RGB 相机:  encoding 通常 'rgb8'  -> (H, W, 3) uint8
  - 深度相机:  encoding 通常 '32FC1' -> (H, W)    float32 (单位: 米)

用法:
    from drivers.camera import CameraHub
    hub = CameraHub({"head_rgb": "r6ef2dc_tp_cam_303d2b1ce0"})
    frame = hub.get_frame("head_rgb")   # None 表示还没收到帧
    hub.shutdown()
"""

import threading

import numpy as np


def image_to_numpy(msg) -> np.ndarray:
    """把 sensor_msgs/Image 解码成 numpy 数组。

    支持的 encoding:
      - rgb8 / bgr8 / rgba8 / mono8  -> uint8
      - 32FC1                        -> float32 (深度图, 单位米)
      - 16UC1                        -> uint16 (深度图, 单位毫米, 需 /1000)
    未知 encoding 退化为 uint8 一维数组(不 reshape), 避免崩溃。
    """
    enc = msg.encoding
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "mono8": 1}
    if enc in channels:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels[enc]
        )
        if enc == "bgr8":
            arr = arr[..., ::-1].copy()  # bgr8 -> rgb8
        return arr
    if enc == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    if enc == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    # 未知编码: 按 uint8 读, 不 reshape
    return np.frombuffer(msg.data, dtype=np.uint8)


class CameraHub:
    """订阅多路相机 topic, 缓存每路最新帧。"""

    def __init__(self, cameras: dict, node_name: str = "camera_hub"):
        """cameras: {逻辑名: topic 名}。"""
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)
        self._frames: dict = {}
        self._lock = threading.Lock()

        for name, topic in cameras.items():
            self.node.create_subscription(
                Image, topic, self._make_cb(name), qos_profile_sensor_data
            )

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _make_cb(self, name):
        def cb(msg):
            arr = image_to_numpy(msg)
            with self._lock:
                self._frames[name] = {
                    "array": arr,
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": msg.encoding,
                    "step": msg.step,
                    "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    "frame_id": msg.header.frame_id,
                }

        return cb

    def get_frame(self, name):
        """返回某一路最新帧(dict 或 None)。"""
        with self._lock:
            return self._frames.get(name)

    def get_all(self):
        """返回 {name: frame} 快照。"""
        with self._lock:
            return dict(self._frames)

    def shutdown(self):
        self._executor.shutdown()
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
