"""PickPlaceRunner:在 LinkerArmA7 + LinkerHandO6Left 上封装 pick/place 原语。

────────────────────────────────────────────────────────────────────────
约定:
  - 每个 move_to 之前先 pose_check,不可达直接放弃,不再尝试(避免无效 IK)。
  - IKSolutionError / JointLimitError / RobocapError 一律捕获并按
    MOTION_RETRIES 重试,超过 raise MotionError 给 __init__.run() 处理。
  - 抓取:hand.grasp_force(strength, fingers) — 该方法在 LinkerHandO6Left 上存在。
    (arm_hand_demo 里的 hand.clench(...) 是过期方法,会 AttributeError。)
  - 释放:hand.open(blocking=True)。
────────────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

logger = logging.getLogger("nut_picker.motion")


class MotionError(RuntimeError):
    """单步动作重试耗尽后抛出。"""


class PickPlaceRunner:
    """封装 pick / place / home / shutdown。"""

    def __init__(self, arm, hand, cfg):
        self.arm = arm
        self.hand = hand
        self.cfg = cfg

    # ── 内部辅助 ─────────────────────────────────────────────────────
    def _pose_check(self, x: float, y: float, z: float) -> bool:
        """调 arm.pose_check,记录失败原因。"""
        try:
            ok, msg = self.arm.pose_check(
                x, y, z,
                roll=self.cfg.GRIPPER_ROLL,
                pitch=self.cfg.GRIPPER_PITCH,
                yaw=self.cfg.GRIPPER_YAW,
            )
        except Exception as e:
            logger.warning(f"[motion] pose_check 抛异常: {e}")
            return False
        if not ok:
            logger.warning(f"[motion] pose_check 不可达 @({x:.3f},{y:.3f},{z:.3f}): {msg}")
        return bool(ok)

    def _move_to(self, x: float, y: float, z: float, label: str) -> bool:
        """带重试的 move_to,异常按 MOTION_RETRIES 次数重试。"""
        last_err = None
        for attempt in range(1, self.cfg.MOTION_RETRIES + 1):
            try:
                self.arm.move_to(
                    x, y, z,
                    roll=self.cfg.GRIPPER_ROLL,
                    pitch=self.cfg.GRIPPER_PITCH,
                    yaw=self.cfg.GRIPPER_YAW,
                    blocking=True,
                )
                return True
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[motion] {label} attempt {attempt}/{self.cfg.MOTION_RETRIES} 失败: {e}"
                )
        raise MotionError(f"{label} 重试 {self.cfg.MOTION_RETRIES} 次仍失败: {last_err}")

    # ── 高层动作 ─────────────────────────────────────────────────────
    def home(self) -> bool:
        """回到 home 姿态 + 张开手。"""
        try:
            self.arm.home(blocking=True)
            self.hand.open(blocking=True)
            return True
        except Exception as e:
            logger.warning(f"[motion] home 失败: {e}")
            return False

    def pick(self, x: float, y: float, z: float) -> bool:
        """在 (x, y, z) 抓一个目标。z 是从 depth 反投影到 base 系的高度(米)。

        流程:approach(抬升到 z+APPROACH) → descend(降到 z+GRASP) →
              grasp_force → lift(回 approach)。
        GRASP_HEIGHT_M 是 z 之上的小偏移(避免压桌面)。
        """
        approach_z = z + self.cfg.APPROACH_HEIGHT_M
        grasp_z = z + self.cfg.GRASP_HEIGHT_M

        if not self._pose_check(x, y, approach_z):
            return False
        self._move_to(x, y, approach_z, f"pick approach @({x:.3f},{y:.3f},{approach_z:.3f})")

        if not self._pose_check(x, y, grasp_z):
            return False
        self._move_to(x, y, grasp_z, f"pick descend @({x:.3f},{y:.3f},{grasp_z:.3f})")

        # 抓取:grasp_force 返回 False 表示力矩未达阈值,但螺母可能已经握住,
        # 不立即 abort,继续 lift + place,看末端是否成功带走。
        grasped = self.hand.grasp_force(
            strength=self.cfg.GRASP_FORCE,
            fingers=self.cfg.GRASP_FINGERS,
            blocking=True,
        )
        logger.info(f"[motion] grasp_force 返回 {grasped}")

        self._move_to(x, y, approach_z, f"pick lift @({x:.3f},{y:.3f},{approach_z:.3f})")
        return True

    def place(self, x: float, y: float, z: float) -> bool:
        """在 (x, y, z) 释放目标。z 是 cell 中心/底面在 base 系的高度(米)。

        流程:approach(抬升到 z+APPROACH) → descend(降到 z+PLACE) →
              open → lift(回 approach)。
        PLACE_HEIGHT_M 是 z 之上的小偏移(避免撞 cell 底)。
        """
        approach_z = z + self.cfg.APPROACH_HEIGHT_M
        place_z = z + self.cfg.PLACE_HEIGHT_M

        if not self._pose_check(x, y, approach_z):
            return False
        self._move_to(x, y, approach_z, f"place approach @({x:.3f},{y:.3f},{approach_z:.3f})")

        if not self._pose_check(x, y, place_z):
            return False
        self._move_to(x, y, place_z, f"place descend @({x:.3f},{y:.3f},{place_z:.3f})")

        self.hand.open(blocking=True)
        logger.info("[motion] hand.open")

        self._move_to(x, y, approach_z, f"place lift @({x:.3f},{y:.3f},{approach_z:.3f})")
        return True

    def shutdown(self) -> None:
        """关掉 arm/hand,异常吞掉。"""
        for name, obj in (("arm", self.arm), ("hand", self.hand)):
            try:
                obj.shutdown()
            except Exception as e:
                logger.warning(f"[motion] {name}.shutdown 异常: {e}")