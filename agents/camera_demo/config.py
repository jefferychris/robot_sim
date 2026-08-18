"""camera_demo 的相机 topic 配置(来自场景传感器 topic)。"""

# 逻辑名 -> ROS2 topic
CAMERAS = {
    "head_rgb": "r6ef2dc_tp_cam_303d2b1ce0",          # base 头部 RGB 相机 1920x1080
    "head_depth": "r6ef2dc_tp_dcam_861ff01d8c",       # base 头部深度相机 640x360
    "left_hand_rgb": "rbd03eb_tp_cam_069a6739f3",     # 左臂末端 L8 RGB 960x540
    "left_hand_depth": "rbd03eb_tp_dcam_3d5a0139cf",  # 左臂末端 L8 深度 320x180
    "right_hand_rgb": "r412d23_tp_cam_3c67aef2bc",    # 右臂末端 L8 RGB 960x540
    "right_hand_depth": "r412d23_tp_dcam_a7a6349c11", # 右臂末端 L8 深度 320x180
}

# 等待首帧的最长时间(秒)
WAIT_TIMEOUT = 20.0

# 保存帧图片的目录(相对工作区根目录)
SAVE_DIR = "camera_frames"
