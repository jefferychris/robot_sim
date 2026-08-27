"""自标定 agent:用机械臂自身当标定物,解 head 相机 → base 的外参。

════════ 为什么要这么做 ════════
现有 HEAD_TO_BASE_T 是拿 gt.json 的 4 个点(3 螺母 + 1 箱子)Kabsch 拟合的。
这 4 点几乎共面(Z 差 < 2cm),旋转分量极不稳定 —— 误差 43.7mm,换位置就失准。

其它路径均已验证不通:
  · 场景 JSON 只给 相机→obzg_33,缺 obzg_33→base_link,链条断裂
  · TF 树实测为空,平台不发布
  · 方像素约束(53.5mm)/残差修正(LOO 417mm)/俯仰约束(96.9mm) 全部更差

本方案让机械臂自己当标定靶:
  move_joints(q) 移动到一批关节位姿 → get_pose() 读回**机器人自报的 base 坐标**
  (绝对可信) → 同时头部相机拍照 → 在深度图中定位末端 → 得到 (base ↔ 像素+深度) 对。
点数任意多、Z 可充分分散,从根上解决 4 点共面的问题。

SDK 事实(tools/scan_sdk.py 实测):
  REMOTE_SERVICES = {pose_check, get_joint_angles, is_moving, get_pose, stop}
  REMOTE_ACTIONS  = {move_to, move_joints, home}
  无 fk/ik/tool_frame 服务,但 DH_PARAMS / JOINT_LIMITS / JOINT_SIGN 在类上可读。
  move_to/pose_check 的默认 pitch 本就是 π —— config 的 GRIPPER_PITCH 并非笔误。

════════ 用法 ════════
在平台点「启动仿真」并勾选「启动当前智能体工程」即可(main.py 的
DEFAULT_AGENT 已指向 self_calib);日志同时写到 camera_frames/self_calib.log。
"""

import logging
import os
import time

import numpy as np

from drivers.camera import CameraHub
from rabo_robocap import LinkerArmA7

from agents.camera_demo.config import CAMERAS as _ALL

__all__ = ["run"]

LOG_PATH = "camera_frames/self_calib.log"
CAMERAS = {k: v for k, v in _ALL.items() if k.startswith("head_")}
ARM_ID = "rbd03ebf4ebf83c6a6a64754454bc520a"      # 左臂
WAIT_FIRST_FRAME = 20.0
SETTLE_S = 1.2                                    # 到位后等画面稳定

logger = logging.getLogger("self_calib")


def _setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")):
        h.setFormatter(fmt)
        root.addHandler(h)
    logger.info("日志同时写入 %s", LOG_PATH)


def _poses():
    """一批关节位姿,让末端在工作空间里铺开(尤其让 Z 分散)。

    以 home(全零)为基准逐关节偏置。JOINT_LIMITS 已知,取值都在限位内。
    """
    base = [0.0] * 7
    out = [list(base)]
    for j, deltas in {
        1: (-0.5, -0.9, -1.3),        # 肩 pitch:主导末端高低 → Z 分散
        3: (-0.4, -0.8, -1.2),        # 肘:主导前后伸展
        0: (-0.4, 0.4),               # 肩 roll:主导左右
        2: (-0.5, 0.5),
        5: (-0.5, 0.5),
    }.items():
        for d in deltas:
            q = list(base)
            q[j] = d
            out.append(q)
    # 组合位姿,进一步分散
    for a, b in ((-0.6, -0.5), (-1.0, -0.8), (-0.8, -1.1), (-1.2, -0.6)):
        q = list(base)
        q[1], q[3] = a, b
        out.append(q)
    return out


def _find_tool_px(depth, prev_depth):
    """用「移动前后深度差」定位末端:变化最大的连通区域即手臂所在。

    比在 RGB 里找特征稳:场景静止,只有手臂在动,差分天然把它分离出来。
    返回 (u, v, z) —— depth 图坐标与该点深度(米)。
    """
    if prev_depth is None or prev_depth.shape != depth.shape:
        return None
    d = np.abs(depth.astype(np.float64) - prev_depth.astype(np.float64))
    d[~np.isfinite(d)] = 0.0
    if d.max() < 0.02:                            # 变化太小,认为没动
        return None
    mask = d > max(0.02, d.max() * 0.35)
    if mask.sum() < 25:
        return None
    try:
        import cv2
        m8 = mask.astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m8, 8)
        if n <= 1:
            return None
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        u, v = cent[i]
        sel = (lab == i) & np.isfinite(depth) & (depth > 0)
        if sel.sum() < 10:
            return None
        return float(u), float(v), float(np.median(depth[sel]))
    except ImportError:
        vs, us = np.nonzero(mask)
        u, v = float(us.mean()), float(vs.mean())
        sel = mask & np.isfinite(depth) & (depth > 0)
        return u, v, float(np.median(depth[sel]))


def _kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def _solve(samples, Wd, Hd):
    """在方像素假设下搜 f,配 Kabsch 解 T。返回 (rmse, f, T, 逐点误差)。"""
    Q = np.array([s["base"] for s in samples])
    cx, cy = Wd / 2.0, Hd / 2.0
    best = None
    for f in np.arange(200.0, 1400.0, 0.5):
        P = np.array([[(s["u"] - cx) * s["z"] / f,
                       (s["v"] - cy) * s["z"] / f,
                       s["z"]] for s in samples])
        R, t = _kabsch(P, Q)
        e = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        rmse = float(np.sqrt((e ** 2).mean()))
        if best is None or rmse < best[0]:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            best = (rmse, f, T, e)
    return best


def run():
    _setup_logging()
    logger.info("=" * 68)
    logger.info("自标定开始 —— 用机械臂当标定靶,解 head→base 外参")
    logger.info("=" * 68)

    hub = CameraHub(CAMERAS, node_name="self_calib_cam")
    deadline = time.time() + WAIT_FIRST_FRAME
    while time.time() < deadline and "head_depth" not in hub.get_all():
        time.sleep(0.2)
    if "head_depth" not in hub.get_all():
        logger.error("未收到 head_depth,退出。确认仿真已启动。")
        hub.shutdown()
        return None
    Hd, Wd = hub.get_frame("head_depth")["array"].shape
    logger.info("深度图 %dx%d", Wd, Hd)

    arm = LinkerArmA7(robot_id=ARM_ID, mode="sim")
    logger.info("DOF=%s  JOINT_LIMITS=%s", arm.DOF, arm.JOINT_LIMITS)

    samples = []
    prev_depth = None
    poses = _poses()
    logger.info("共 %d 个位姿待采集", len(poses))

    for i, q in enumerate(poses):
        logger.info("-" * 68)
        logger.info("[%2d/%d] move_joints %s", i + 1, len(poses),
                    [round(v, 2) for v in q])
        try:
            arm.move_joints(q, blocking=True)
        except Exception as e:
            logger.warning("     移动失败,跳过: %s", str(e)[:90])
            continue
        time.sleep(SETTLE_S)

        try:
            pos, rpy = arm.get_pose()
            angles = arm.get_joint_angles()
        except Exception as e:
            logger.warning("     读位姿失败: %s", str(e)[:90])
            continue

        fr = hub.get_frame("head_depth")
        depth = fr["array"].copy() if fr else None
        if depth is None:
            logger.warning("     无深度帧")
            continue

        logger.info("     末端 base=(%.4f, %.4f, %.4f)  rpy=(%.3f, %.3f, %.3f)",
                    *pos, *rpy)
        logger.info("     关节实际=%s", [round(a, 3) for a in angles])

        hit = _find_tool_px(depth, prev_depth)
        prev_depth = depth
        if hit is None:
            logger.info("     深度差分未定位到末端(首帧或位移过小),跳过")
            continue
        u, v, z = hit
        logger.info("     depth_px=(%.1f, %.1f)  z=%.4f m  ✓ 采纳", u, v, z)
        samples.append({"base": np.array(pos, float), "u": u, "v": v, "z": z})

    try:
        arm.home(blocking=True)
        arm.shutdown()
    except Exception:
        pass
    hub.shutdown()

    logger.info("=" * 68)
    logger.info("采集到 %d 个有效样本", len(samples))
    if len(samples) < 6:
        logger.error("样本不足(<6),无法可靠标定。")
        return None

    zs = [s["base"][2] for s in samples]
    logger.info("末端 Z 分布: %.3f ~ %.3f (跨度 %.3f m)  ← 越大越好",
                min(zs), max(zs), max(zs) - min(zs))

    rmse, f, T, err = _solve(samples, Wd, Hd)
    logger.info("=" * 68)
    logger.info("标定结果: f=%.2f   RMSE=%.1f mm    [旧 4 点方案: 43.7mm]", f, rmse * 1000)
    for s, e in zip(samples, err):
        logger.info("   base=(%7.4f,%7.4f,%7.4f)  err=%6.1f mm", *s["base"], e * 1000)

    logger.info("=" * 68)
    logger.info("# 贴进 agents/nut_picker/config.py")
    logger.info('HEAD_CAMERA_INTRINSICS = {"fx": %.2f, "fy": %.2f, "cx": %.2f, "cy": %.2f}',
                f, f, Wd / 2.0, Hd / 2.0)
    logger.info("HEAD_TO_BASE_T = np.array([")
    for row in T:
        logger.info("    [%s],", ", ".join("% .8f" % v for v in row))
    logger.info("], dtype=np.float64)")
    logger.info("=" * 68)
    logger.info("完整日志: %s", LOG_PATH)
    return T
