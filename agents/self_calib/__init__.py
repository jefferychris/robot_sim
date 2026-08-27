"""自标定 agent:用机械臂自身当标定靶,解 head 相机 → base 的外参。

════════ 为什么需要它 ════════
现有 HEAD_TO_BASE_T 来自 gt.json 的 4 点 Kabsch 拟合。4 点几乎共面
(Z 跨度 < 2cm),旋转分量不稳 —— 误差 43.7mm 且换位置就失准。
其余路径均已实测排除:场景 JSON 缺 obzg_33→base_link;TF 树为空;
方像素 53.5mm;残差修正 LOO 417mm;俯仰约束 96.9mm。

本方案让机械臂当标定靶:move_joints 走一批位姿 → get_pose() 读回
**机器人自报的 base 坐标**(绝对可信)→ 同时拍深度图 → 帧差定位末端
→ 得 (base ↔ 像素+深度) 对。点数任意多、Z 可充分分散。

════════ 上一轮(v1)的教训 ════════
17 个位姿全部"未定位到末端"。原因:v1 用肩 pitch 负向偏置,实测末端
Y 从 0.2035 一路涨到 0.8117 —— 而用户观察到手臂"背对桌子"运动。
=> **Y 正方向背对桌子,桌子在 Y 负侧**;手臂全程在相机视野外,自然拍不到。
(顺带:视觉给的螺母 Y=-0.067/-0.168/-0.090 符号其实是对的。)
另有若干位姿撞到桌子,故本版加 pose_check 预检 + Z 下限保护。

位姿设计沿用 yizhi-robot/src_remote/auto_calibrate.py 的分布原则
(多高度分层 + XY 铺开 + 姿态偏转),但数值按仿真实测可达范围重定。

════════ 用法 ════════
点「启动仿真」并勾选「启动当前智能体工程」,或终端 python main.py。
先跑 PROBE 阶段(6 点,慢速,确认能拍到手臂),再自动进入全量采集。
日志落盘 camera_frames/logs/self_calib_<时间戳>.log(保留历史),
最近一次镜像到 camera_frames/self_calib_latest.log;
每帧深度图存 camera_frames/calib_<时间戳>_*.npy。
"""

import logging
import os
import time

import numpy as np

from drivers.camera import CameraHub
from rabo_robocap import LinkerArmA7

from agents.camera_demo.config import CAMERAS as _ALL

__all__ = ["run"]

FRAME_DIR = "camera_frames"
LOG_DIR = "camera_frames/logs"
# 每次运行写一个带时间戳的新文件,不覆盖历史(前后对比是排查的主要手段);
# 同时把最近一次复制成 latest,方便直接 cat。
LATEST_LOG = "camera_frames/self_calib_latest.log"
CAMERAS = {k: v for k, v in _ALL.items() if k.startswith("head_")}
ARM_ID = "rbd03ebf4ebf83c6a6a64754454bc520a"      # 左臂
WAIT_FIRST_FRAME = 20.0
SETTLE_S = 1.5

# 安全:末端 Z 不低于此值,避免撞桌(home 在 -0.6082,桌面更低)
Z_FLOOR = -0.62
# 帧差判定阈值
MIN_DEPTH_DELTA = 0.015
MIN_BLOB_PX = 20

logger = logging.getLogger("self_calib")
RUN_STAMP = time.strftime("%Y%m%d-%H%M%S")


def _setup_logging():
    """日志写到带时间戳的新文件(保留历史),并镜像一份到 latest。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    run_log = os.path.join(LOG_DIR, "self_calib_%s.log"
                           % time.strftime("%Y%m%d-%H%M%S"))
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in (logging.StreamHandler(),
              logging.FileHandler(run_log, mode="w"),
              logging.FileHandler(LATEST_LOG, mode="w")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return run_log


def _probe_poses():
    """6 个探测位姿:朝 Y 负(桌子侧)小幅试探,确认相机能拍到手臂。"""
    return [
        [0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0],   # home 基准帧
        [0.0,  0.3,  0.0, -0.3, 0.0, 0.0, 0.0],
        [0.0,  0.5,  0.0, -0.5, 0.0, 0.0, 0.0],
        [-0.3, 0.4,  0.0, -0.4, 0.0, 0.0, 0.0],
        [0.3,  0.4,  0.0, -0.4, 0.0, 0.0, 0.0],
        [0.0,  0.6,  0.0, -0.7, 0.0, 0.0, 0.0],
    ]


def _full_poses():
    """全量位姿:在 Y 负侧(桌子方向)分层铺开。

    沿用 yizhi auto_calibrate 的分布原则:多高度 + XY 铺开 + 姿态偏转,
    数值按仿真实测重定。肩 pitch 取**正值**才朝桌子(v1 取负值是反的)。
    """
    out = []
    for sh in (0.3, 0.45, 0.6, 0.75):              # 肩 pitch:主导前伸与高度
        for el in (-0.3, -0.5, -0.7):              # 肘:调节远近
            out.append([0.0, sh, 0.0, el, 0.0, 0.0, 0.0])
    for roll in (-0.4, -0.2, 0.2, 0.4):            # 肩 roll:左右铺开
        out.append([roll, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])
    for yaw in (-0.4, 0.4):                        # 肩 yaw
        out.append([0.0, 0.5, yaw, -0.5, 0.0, 0.0, 0.0])
    for wr in (-0.4, 0.4):                         # 腕:小幅改变末端朝向
        out.append([0.0, 0.5, 0.0, -0.5, 0.0, wr, 0.0])
    return out


def _find_tool_px(depth, ref_depth):
    """用「与基准帧的深度差」定位末端:场景静止,只有手臂在动。"""
    if ref_depth is None or ref_depth.shape != depth.shape:
        return None
    d = np.abs(depth.astype(np.float64) - ref_depth.astype(np.float64))
    d[~np.isfinite(d)] = 0.0
    if d.max() < MIN_DEPTH_DELTA:
        return None, d.max()
    mask = d > max(MIN_DEPTH_DELTA, d.max() * 0.35)
    if mask.sum() < MIN_BLOB_PX:
        return None, d.max()
    try:
        import cv2
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n <= 1:
            return None, d.max()
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        u, v = cent[i]
        sel = (lab == i) & np.isfinite(depth) & (depth > 0)
        if sel.sum() < 10:
            return None, d.max()
        return (float(u), float(v), float(np.median(depth[sel])),
                int(stats[i, cv2.CC_STAT_AREA])), d.max()
    except ImportError:
        vs, us = np.nonzero(mask)
        sel = mask & np.isfinite(depth) & (depth > 0)
        return (float(us.mean()), float(vs.mean()),
                float(np.median(depth[sel])), int(mask.sum())), d.max()


def _kabsch(P, Q):
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def _solve(samples, Wd, Hd):
    Q = np.array([s["base"] for s in samples])
    cx, cy = Wd / 2.0, Hd / 2.0
    best = None
    for f in np.arange(200.0, 1400.0, 0.5):
        P = np.array([[(s["u"] - cx) * s["z"] / f,
                       (s["v"] - cy) * s["z"] / f, s["z"]] for s in samples])
        R, t = _kabsch(P, Q)
        e = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        rmse = float(np.sqrt((e ** 2).mean()))
        if best is None or rmse < best[0]:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            best = (rmse, f, T, e)
    return best


def _collect(arm, hub, poses, ref_depth, label, samples):
    """跑一批位姿,采样并落盘深度图。返回更新后的 ref_depth。"""
    for i, q in enumerate(poses):
        logger.info("-" * 68)
        logger.info("[%s %2d/%d] move_joints %s", label, i + 1, len(poses),
                    [round(v, 2) for v in q])
        try:
            arm.move_joints(q, blocking=True)
        except Exception as e:
            logger.warning("     移动失败: %s", str(e)[:90])
            continue
        time.sleep(SETTLE_S)

        try:
            pos, rpy = arm.get_pose()
        except Exception as e:
            logger.warning("     读位姿失败: %s", str(e)[:90])
            continue

        if pos[2] < Z_FLOOR:
            logger.warning("     末端 Z=%.4f 低于安全下限 %.2f,跳过采样", pos[2], Z_FLOOR)
            continue

        fr = hub.get_frame("head_depth")
        if fr is None:
            logger.warning("     无深度帧")
            continue
        depth = fr["array"].copy()

        logger.info("     末端 base=(%.4f, %.4f, %.4f)  rpy=(%.3f, %.3f, %.3f)",
                    *pos, *rpy)

        np.save(os.path.join(FRAME_DIR, "calib_%s_%s_%02d.npy"
                             % (RUN_STAMP, label, i)), depth)

        if ref_depth is None:
            ref_depth = depth
            logger.info("     ← 设为基准帧(home 姿态,手臂不在桌面视野)")
            continue

        hit, dmax = _find_tool_px(depth, ref_depth)
        if hit is None:
            logger.info("     未定位到末端 (最大深度变化 %.4f m,阈值 %.3f)",
                        dmax, MIN_DEPTH_DELTA)
            continue
        u, v, z, area = hit
        logger.info("     depth_px=(%.1f, %.1f)  z=%.4f m  area=%d px  ✓ 采纳",
                    u, v, z, area)
        samples.append({"base": np.array(pos, float), "u": u, "v": v, "z": z})
    return ref_depth


def run():
    run_log = _setup_logging()
    logger.info("=" * 68)
    logger.info("自标定 v2 —— Y 负侧(桌子方向)采样,带撞桌保护")
    logger.info("本次日志: %s  (最近一次镜像: %s)", run_log, LATEST_LOG)
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
    logger.info("深度图 %dx%d;末端 Z 安全下限 %.2f", Wd, Hd, Z_FLOOR)

    arm = LinkerArmA7(robot_id=ARM_ID, mode="sim")
    samples = []

    # ── 阶段一:探测 ────────────────────────────────────────────────
    logger.info("\n【阶段一】探测 6 点,确认相机能拍到手臂")
    ref = _collect(arm, hub, _probe_poses(), None, "probe", samples)
    logger.info("=" * 68)
    logger.info("探测阶段采到 %d 个样本", len(samples))

    if not samples:
        logger.error("探测阶段一个都没采到 —— 手臂可能仍在相机视野外。")
        logger.error("深度图已存 camera_frames/calib_%s_probe_*.npy,可离线分析。",
                     RUN_STAMP)
        try:
            arm.home(blocking=True)
            arm.shutdown()
        except Exception:
            pass
        hub.shutdown()
        return None

    # ── 阶段二:全量 ────────────────────────────────────────────────
    logger.info("\n【阶段二】全量采集")
    _collect(arm, hub, _full_poses(), ref, "full", samples)

    try:
        arm.home(blocking=True)
        arm.shutdown()
    except Exception:
        pass
    hub.shutdown()

    logger.info("=" * 68)
    logger.info("共采集 %d 个有效样本", len(samples))
    if len(samples) < 6:
        logger.error("样本不足(<6),无法可靠标定。")
        return None

    for ax, name in enumerate("XYZ"):
        vals = [s["base"][ax] for s in samples]
        logger.info("末端 %s 分布: %.3f ~ %.3f (跨度 %.3f m)",
                    name, min(vals), max(vals), max(vals) - min(vals))
    logger.info("  ← gt.json 4 点的 Z 跨度仅 0.006m,那是旧标定失准的根源")

    rmse, f, T, err = _solve(samples, Wd, Hd)
    logger.info("=" * 68)
    logger.info("标定结果: f=%.2f   RMSE=%.1f mm    [旧 4 点方案 43.7mm]",
                f, rmse * 1000)
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
    return T
