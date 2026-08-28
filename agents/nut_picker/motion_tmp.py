"""【临时预写 · 未校验】基于 move_joints 的抓放逻辑。

╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  本文件为预写稿,尚未在仿真环境中跑通验证。                        ║
║                                                                      ║
║  状态: 逻辑完整,但所有关节角常量均为占位值(全 0 / 猜测值),           ║
║        必须先做示教采点填入 TEACH_POINTS 才能实际运行。               ║
║                                                                      ║
║  为什么单独开一个文件:                                                ║
║    - 与现役 motion.py 完全解耦,互不影响,现有流程不受破坏             ║
║    - _tmp 后缀标明这是临时件,验证通过后再合并回 motion.py            ║
║    - 现在不被 __init__.py 引用,不会误跑                              ║
╚══════════════════════════════════════════════════════════════════════╝

──────────────────────────────────────────────────────────────────────
背景:为什么不能用 move_to
──────────────────────────────────────────────────────────────────────
2026-08-28 用 tools/probe_workspace.py 实测:

  桌面高度 z=-0.47 时,三颗螺母在**所有姿态**下 pose_check 均返回
  out_of_workspace;把 z 抬到 -0.10 以上才可达。

  但机械臂 home 位末端就在 z=-0.6082 —— 比桌面还低 14cm,物理上显然够得到。

  结论:限制来自 move_to 内部的 IK 解算器(可能只解特定构型 / 内部限了关节范围),
  不是几何不可达。故笛卡尔路径走不通,必须走关节空间 move_joints。

──────────────────────────────────────────────────────────────────────
本文件的方案:示教点 + 雅可比增量修正
──────────────────────────────────────────────────────────────────────
螺母位置每局会变,不能录死关节角。所以:

  1. 【离线一次】人工遥控把臂摆到"末端悬停于桌面某点正上方、指尖朝下"的姿势,
     记下 (关节角 q, 末端XYZ p)。在桌面不同位置采 ≥4 个点 → TEACH_POINTS。

  2. 【离线一次】由这些点数值估计雅可比 J = ∂p_xy / ∂q (3×7 取前两行),
     得到"末端 XY 位移 → 关节角增量"的线性映射 → JACOBIAN_INV。

  3. 【每局运行】视觉给出螺母 XY(Z 用固定桌面高度 config.TABLE_Z_M):
       - 选最近的示教点作种子 q_seed
       - Δp = 目标XY - 种子XY
       - q_target = q_seed + JACOBIAN_INV @ Δp
       - move_joints(q_target)

  线性化只在种子点邻域成立,所以示教点要铺开覆盖桌面,让 Δp 保持小(< ~8cm)。

──────────────────────────────────────────────────────────────────────
待办(按顺序)
──────────────────────────────────────────────────────────────────────
  [ ] 采集 TEACH_POINTS —— 见文件末尾 collect_teach_point() 用法
  [ ] 由示教点算 JACOBIAN_INV —— 见 fit_jacobian()
  [ ] 验证 solve_joints() 精度(反解后 move_joints,比对实际末端 XY)
  [ ] 标定 GRASP_CLEARANCE_M(螺母顶面到桌面的高度,M27/M33/M45 各不同)
  [ ] 标定手部 grasp_force strength 对不同螺母的合适值
  [ ] 全流程连跑,统计成功率
──────────────────────────────────────────────────────────────────────
"""

import logging
import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("nut_picker.motion_tmp")

DOF = 7  # LinkerArmA7


class MotionError(RuntimeError):
    """单步动作失败(重试耗尽 / 无解)。"""


# ══════════════════════════════════════════════════════════════════════
#  待标定常量  ← 全部是占位值,必须实测填入
# ══════════════════════════════════════════════════════════════════════

# ── 示教点:(关节角 7 维, 末端 XYZ) ────────────────────────────────────
# 采集方式:遥控器摆到「末端悬停于桌面上方、指尖朝下」,跑 collect_teach_point()。
# 要求:≥4 个点,在桌面 XY 上尽量铺开(覆盖三颗螺母可能出现的区域),
#       且所有点的末端 Z 尽量一致(同一悬停高度),这样雅可比只反映 XY 平面运动。
#
# ⚠️ 下面全是占位符,当前跑会得到无意义结果。
TEACH_POINTS: List[dict] = [
    # {"q": [0.0]*7, "p": (0.35, 0.05, -0.35), "note": "示例:桌面左前"},
    # {"q": [0.0]*7, "p": (0.45, 0.05, -0.35), "note": "示例:桌面右前"},
    # {"q": [0.0]*7, "p": (0.35, 0.20, -0.35), "note": "示例:桌面左后"},
    # {"q": [0.0]*7, "p": (0.45, 0.20, -0.35), "note": "示例:桌面右后"},
]

# ── 雅可比伪逆:末端 XYZ 位移 → 关节角增量,形状 (7, 3) ────────────────
# 由 fit_jacobian(TEACH_POINTS) 算出后填这里,避免每次启动重算。
# None 表示尚未标定,solve_joints() 会退化为「直接用最近示教点的 q」。
JACOBIAN_INV: Optional[np.ndarray] = None

# ── 高度参数(米,SDK base 坐标) ───────────────────────────────────────
# 桌面高度取 config.TABLE_Z_M(-0.47)。以下都是相对它的偏移。
HOVER_HEIGHT_M = 0.12        # 悬停高度:抓取前/后在目标正上方停这么高
GRASP_CLEARANCE_M = 0.015    # 抓取时指尖底面距桌面的高度(≈螺母半高,待分尺寸标定)
LIFT_HEIGHT_M = 0.15         # 抓到后抬升高度
PLACE_CLEARANCE_M = 0.03     # 放入格子时的释放高度(格子底面之上)

# ── 手部参数 ─────────────────────────────────────────────────────────
GRASP_STRENGTH = 60          # grasp_force strength,0~100,待按螺母重量调
GRASP_FINGERS = [0, 1, 2, 3, 4]
HAND_SETTLE_S = 0.5          # 手部动作后的稳定等待
HAND_CMD_REPEAT = 2          # 手指指令重发次数

# 分尺寸抓握力度(占位值,待标定)。
# 参考 yizhi-robot 项目经验:三种螺母必须用三种不同闭合度,统一手型抓不稳。
# 那边(LinkerHand O6,6指 0~255 制)实测值为:
#   large  [128,0,127,128,110,100] / medium [125,0,100,151,33,31] / small [128,0,35,128,28,31]
# 本项目 LinkerHandO6 走 grasp_force(strength) 接口,量纲不同,仅借策略。
GRASP_STRENGTH_BY_LABEL = {
    "big":    55,            # M45 最大,力度可略小(接触面大)
    "medium": 60,
    "small":  70,            # M27 最小,需更大闭合度才夹得住
}

# ── 提起降级高度 ─────────────────────────────────────────────────────
# 抓到后抬升可能因 IK 无解失败。逐级退让而非直接放弃(yizhi 项目实证有效)。
LIFT_FALLBACKS_M = [0.15, 0.13, 0.10, 0.07]

# ── 位姿校验 ─────────────────────────────────────────────────────────
# 每步动作后读回实际末端位置,偏差超阈值则重试(yizhi 项目实证有效)。
POSE_VERIFY_TOL_M = 0.01     # 1cm 容差
POSE_VERIFY = True

# ── 运动参数 ─────────────────────────────────────────────────────────
MOVE_SETTLE_S = 0.3          # 每步 move_joints 后的稳定等待
MOTION_RETRIES = 2
MAX_EXTRAPOLATION_M = 0.10   # Δp 超过这个距离就拒绝(线性化失效),报错而非乱动


# ══════════════════════════════════════════════════════════════════════
#  标定工具(离线跑一次)
# ══════════════════════════════════════════════════════════════════════

def collect_teach_point(arm, note: str = "") -> dict:
    """读当前臂姿,返回可直接粘进 TEACH_POINTS 的一条记录。

    用法(平台终端):
        1. 先用遥控器把右臂摆到目标姿势(末端悬于桌面上方、指尖朝下)
        2. 跑:
            from rabo_robocap import LinkerArmA7
            from agents.nut_picker import motion_tmp as m
            arm = LinkerArmA7(robot_id="r412d237980e3167577d7aece10f7aedb", mode="sim")
            print(m.collect_teach_point(arm, note="桌面左前"))
        3. 把打印出的 dict 粘进本文件的 TEACH_POINTS
        4. 换个位置重复,共采 ≥4 个点
    """
    q = list(arm.get_joint_angles())
    pos, rpy = arm.get_pose()
    rec = {"q": [round(v, 6) for v in q],
           "p": tuple(round(v, 6) for v in pos),
           "rpy": tuple(round(v, 6) for v in rpy),
           "note": note}
    logger.info(f"[teach] 采点: {rec}")
    return rec


def fit_jacobian(points: Sequence[dict]) -> Optional[np.ndarray]:
    """由示教点最小二乘拟合「末端位移 → 关节增量」的线性映射。

    做法:以第一个点为基准,构造 ΔP (N-1, 3) 和 ΔQ (N-1, 7),
    解 ΔQ ≈ ΔP @ M,其中 M 形状 (3, 7);返回 M.T 即 (7, 3)。

    返回 None 表示点数不足(需 ≥4 个,才能定 3 个平移自由度并留冗余)。
    """
    if len(points) < 4:
        logger.warning(f"[jacobian] 示教点仅 {len(points)} 个,需 ≥4,跳过拟合")
        return None

    base = points[0]
    q0 = np.asarray(base["q"], dtype=float)
    p0 = np.asarray(base["p"], dtype=float)

    dP = np.array([np.asarray(pt["p"], float) - p0 for pt in points[1:]])  # (N-1, 3)
    dQ = np.array([np.asarray(pt["q"], float) - q0 for pt in points[1:]])  # (N-1, 7)

    # 最小二乘: dP @ M = dQ  →  M (3,7)
    M, *_ = np.linalg.lstsq(dP, dQ, rcond=None)
    J_inv = M.T  # (7, 3)

    resid = dP @ M - dQ
    rms = float(np.sqrt((resid ** 2).mean()))
    logger.info(f"[jacobian] 拟合完成,残差 RMS={rms:.5f} rad")
    if rms > 0.05:
        logger.warning("[jacobian] 残差偏大,示教点可能跨度过大或姿态不一致")
    return J_inv


# ══════════════════════════════════════════════════════════════════════
#  关节角求解
# ══════════════════════════════════════════════════════════════════════

def _nearest_seed(x: float, y: float, z: float,
                  points: Sequence[dict]) -> Tuple[dict, float]:
    """挑 XY 距离最近的示教点作种子。"""
    target = np.array([x, y], dtype=float)
    best, best_d = None, float("inf")
    for pt in points:
        d = float(np.linalg.norm(np.asarray(pt["p"][:2], float) - target))
        if d < best_d:
            best, best_d = pt, d
    return best, best_d


def solve_joints(x: float, y: float, z: float,
                 points: Sequence[dict] = None,
                 jacobian_inv: Optional[np.ndarray] = None) -> List[float]:
    """目标末端 XYZ → 关节角(线性化近似)。

    q = q_seed + J⁻¹ · (p_target - p_seed)

    Raises:
        MotionError: 无示教点,或目标离所有示教点都太远(线性化失效)。
    """
    pts = points if points is not None else TEACH_POINTS
    if not pts:
        raise MotionError(
            "TEACH_POINTS 为空 —— 需先示教采点,见 motion_tmp.collect_teach_point()"
        )

    seed, dist = _nearest_seed(x, y, z, pts)
    if dist > MAX_EXTRAPOLATION_M:
        raise MotionError(
            f"目标 ({x:.3f},{y:.3f}) 距最近示教点 {dist:.3f}m > "
            f"{MAX_EXTRAPOLATION_M}m,线性化不可靠。请在该区域补采示教点。"
        )

    q_seed = np.asarray(seed["q"], dtype=float)
    p_seed = np.asarray(seed["p"], dtype=float)
    dp = np.array([x, y, z], dtype=float) - p_seed

    J_inv = jacobian_inv if jacobian_inv is not None else JACOBIAN_INV
    if J_inv is None:
        logger.warning(
            f"[solve] JACOBIAN_INV 未标定,退化为直接用种子关节角 "
            f"(末端将偏离目标 {np.linalg.norm(dp):.3f}m)"
        )
        return q_seed.tolist()

    q = q_seed + J_inv @ dp
    logger.debug(f"[solve] seed={seed.get('note','')} Δp={dp.round(4)} → q={q.round(4)}")
    return q.tolist()


# ══════════════════════════════════════════════════════════════════════
#  抓放执行
# ══════════════════════════════════════════════════════════════════════

class PickPlaceRunnerTmp:
    """走 move_joints 的抓放执行器(临时预写版)。

    与现役 PickPlaceRunner 接口保持一致(home/pick/place/shutdown),
    验证通过后可直接顶替。
    """

    def __init__(self, arm, hand, cfg):
        self.arm = arm
        self.hand = hand
        self.cfg = cfg
        self.table_z = getattr(cfg, "TABLE_Z_M", -0.47)

    # ── 底层 ─────────────────────────────────────────────────────────
    def _move_joints(self, q: Sequence[float], label: str,
                     expect_xyz: Optional[Tuple[float, float, float]] = None) -> bool:
        """带重试的 move_joints,可选末端位姿回读校验。

        expect_xyz 给出时,动作后读回实际末端位置比对;偏差 > POSE_VERIFY_TOL_M
        视为失败并重试(线性化解算在种子点较远处会有偏差,靠这层兜住)。
        """
        if len(q) != DOF:
            raise MotionError(f"关节角维度错误: 期望 {DOF},得到 {len(q)}")
        last_err = None
        for attempt in range(1, MOTION_RETRIES + 1):
            try:
                self.arm.move_joints(list(q), blocking=True)
                time.sleep(MOVE_SETTLE_S)

                if POSE_VERIFY and expect_xyz is not None:
                    err = self._pose_error(expect_xyz)
                    if err is not None and err > POSE_VERIFY_TOL_M:
                        logger.warning(
                            f"[motion] {label} 位姿偏差 {err*1000:.1f}mm "
                            f"> {POSE_VERIFY_TOL_M*1000:.0f}mm,第 {attempt} 次"
                        )
                        last_err = f"位姿偏差 {err*1000:.1f}mm"
                        continue

                logger.info(f"[motion] {label} ok  q={[round(v,3) for v in q]}")
                return True
            except Exception as e:
                last_err = e
                logger.warning(f"[motion] {label} 第 {attempt}/{MOTION_RETRIES} 次失败: {e}")
        logger.error(f"[motion] {label} 重试耗尽: {last_err}")
        return False

    def _pose_error(self, expect_xyz: Tuple[float, float, float]) -> Optional[float]:
        """读回末端实际位置,返回与期望的欧氏距离(米)。读不到返回 None。"""
        try:
            pos, _rpy = self.arm.get_pose()
        except Exception as e:
            logger.debug(f"[motion] get_pose 失败: {e}")
            return None
        d = float(np.linalg.norm(np.asarray(pos, float) - np.asarray(expect_xyz, float)))
        return d

    def _goto_xyz(self, x: float, y: float, z: float, label: str) -> bool:
        """目标 XYZ → 解关节角 → 执行。"""
        try:
            q = solve_joints(x, y, z)
        except MotionError as e:
            logger.error(f"[motion] {label} 解算失败: {e}")
            return False
        return self._move_joints(q, f"{label} @({x:.3f},{y:.3f},{z:.3f})",
                                 expect_xyz=(x, y, z))

    def _grasp(self, label: Optional[str] = None) -> bool:
        """抓紧。label 给出时按螺母尺寸选力度。

        指令重发 HAND_CMD_REPEAT 次 —— 手部状态读不回来,重发保证到达
        (yizhi 项目实证:发一次偶发丢失)。
        """
        strength = GRASP_STRENGTH_BY_LABEL.get(label, GRASP_STRENGTH)
        try:
            for _ in range(HAND_CMD_REPEAT):
                self.hand.grasp_force(strength, GRASP_FINGERS, blocking=True)
            time.sleep(HAND_SETTLE_S)
            logger.info(f"[motion] 抓紧 label={label} strength={strength}")
            return True
        except Exception as e:
            logger.error(f"[motion] grasp_force 失败: {e}")
            return False

    def _release(self) -> bool:
        try:
            for _ in range(HAND_CMD_REPEAT):
                self.hand.open(blocking=True)
            time.sleep(HAND_SETTLE_S)
            logger.info("[motion] 松开")
            return True
        except Exception as e:
            logger.error(f"[motion] hand.open 失败: {e}")
            return False

    def _check_grasped(self) -> Optional[bool]:
        """用指尖力反馈判断是否真的抓住了。

        HandBase.get_joint_forces() 存在。阈值待标定 —— 现在只记录不判定。
        返回 None 表示无法判断。
        """
        try:
            forces = self.hand.get_joint_forces()
        except Exception as e:
            logger.debug(f"[motion] get_joint_forces 不可用: {e}")
            return None
        logger.info(f"[motion] 指尖力: {[round(f,2) for f in forces]}")
        return None  # TODO: 标定力阈值后改成真判定

    # ── 对外原语 ─────────────────────────────────────────────────────
    def home(self) -> bool:
        """回初始位。"""
        try:
            self.arm.home(blocking=True)
            self.hand.open(blocking=True)
            time.sleep(MOVE_SETTLE_S)
            logger.info("[motion] home 完成")
            return True
        except Exception as e:
            logger.error(f"[motion] home 失败: {e}")
            return False

    def _lift_with_fallback(self, x: float, y: float, z_table: float,
                            label: str) -> bool:
        """抬起,高度逐级退让。

        高抬升可能 IK 无解,与其直接失败不如降低要求 —— 抬 7cm 也够挪走。
        (yizhi 项目实证有效)
        """
        for h in LIFT_FALLBACKS_M:
            if self._goto_xyz(x, y, z_table + h, f"{label}(+{h*100:.0f}cm)"):
                return True
            logger.warning(f"[motion] 抬升 {h*100:.0f}cm 失败,降级重试")
        logger.error(f"[motion] {label} 所有降级高度均失败")
        return False

    def pick(self, x: float, y: float, z: float = None,
             label: Optional[str] = None) -> bool:
        """抓取:悬停 → 张手 → 下降 → 抓紧 → 抬起。

        z 传 None 时用固定桌面高度(推荐 —— 视觉只出 XY)。
        label 为 big/medium/small,用于选抓握力度。
        """
        z_table = self.table_z if z is None else z
        z_hover = z_table + HOVER_HEIGHT_M
        z_grasp = z_table + GRASP_CLEARANCE_M

        logger.info(f"[pick] {label or ''} 目标 ({x:.3f},{y:.3f}) 桌面 z={z_table:.3f}")

        if not self._goto_xyz(x, y, z_hover, "pick-悬停"):
            return False
        if not self._release():          # 先张开
            return False
        if not self._goto_xyz(x, y, z_grasp, "pick-下降"):
            return False
        if not self._grasp(label):
            return False
        self._check_grasped()
        if not self._lift_with_fallback(x, y, z_table, "pick-抬起"):
            self._recover()
            return False
        return True

    def _recover(self) -> bool:
        """任何一步失败后的恢复:松手 + 回 home,避免卡在半空挡住视线。"""
        logger.warning("[motion] 进入恢复流程")
        self._release()
        return self.home()

    def place(self, x: float, y: float, z: float = None) -> bool:
        """放置:悬停 → 下降 → 松开 → 抬起。"""
        z_table = self.table_z if z is None else z
        z_hover = z_table + HOVER_HEIGHT_M
        z_place = z_table + PLACE_CLEARANCE_M

        logger.info(f"[place] 目标 ({x:.3f},{y:.3f}) 桌面 z={z_table:.3f}")

        if not self._goto_xyz(x, y, z_hover, "place-悬停"):
            return False
        if not self._goto_xyz(x, y, z_place, "place-下降"):
            return False
        if not self._release():
            return False
        if not self._lift_with_fallback(x, y, z_table, "place-抬起"):
            return False
        return True

    def shutdown(self):
        for obj, name in ((self.arm, "arm"), (self.hand, "hand")):
            try:
                obj.shutdown()
            except Exception as e:
                logger.warning(f"[motion] {name}.shutdown 异常: {e}")


# ══════════════════════════════════════════════════════════════════════
#  自检(不连机器人,验证解算逻辑本身)
# ══════════════════════════════════════════════════════════════════════

def _selftest():
    """用合成数据验证 fit_jacobian + solve_joints 的数学正确性。

    构造一个已知的线性映射,看能不能被拟合出来并正确反解。
    平台终端跑: python3 -m agents.nut_picker.motion_tmp
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    print("=" * 66)
    print("motion_tmp 自检 —— 合成数据验证解算逻辑(不连机器人)")
    print("=" * 66)

    rng = np.random.default_rng(0)
    J_true = rng.normal(0, 1.0, size=(7, 3))     # 真实的 位移→关节 映射
    q0 = rng.normal(0, 0.5, size=7)
    p0 = np.array([0.40, 0.10, -0.35])

    pts = []
    for dx, dy in [(0, 0), (0.06, 0), (0, 0.06), (0.06, 0.06), (-0.05, 0.03)]:
        dp = np.array([dx, dy, 0.0])
        pts.append({"q": (q0 + J_true @ dp).tolist(),
                    "p": tuple(p0 + dp),
                    "note": f"synthetic({dx},{dy})"})
    print(f"\n构造 {len(pts)} 个合成示教点")

    J_fit = fit_jacobian(pts)
    assert J_fit is not None, "拟合应当成功"
    # 只有 XY 方向被激励,Z 列不可辨识,故只比对前两列
    err = np.abs(J_fit[:, :2] - J_true[:, :2]).max()
    print(f"雅可比拟合最大误差(XY 列): {err:.2e}")
    assert err < 1e-6, f"拟合误差过大: {err}"

    tx, ty = 0.43, 0.13
    q = solve_joints(tx, ty, -0.35, points=pts, jacobian_inv=J_fit)
    q_expect = q0 + J_true @ np.array([tx - p0[0], ty - p0[1], 0.0])
    err2 = np.abs(np.asarray(q) - q_expect).max()
    print(f"反解关节角最大误差: {err2:.2e}")
    assert err2 < 1e-6, f"反解误差过大: {err2}"

    # 外推保护
    try:
        solve_joints(0.9, 0.9, -0.35, points=pts, jacobian_inv=J_fit)
        raise AssertionError("远距离目标应当被拒绝")
    except MotionError as e:
        print(f"外推保护生效: {str(e)[:60]}...")

    # 点数不足
    assert fit_jacobian(pts[:2]) is None
    print("点数不足时正确返回 None")

    print("\n✅ 解算逻辑自检全部通过")
    print("⚠️  注意:这只验证了数学,真实关节角/雅可比仍需示教采点标定")


if __name__ == "__main__":
    _selftest()
