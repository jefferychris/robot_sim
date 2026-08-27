"""对照真机 rosbridge API,扫仿真 SDK 与 ROS2 底层暴露了什么。

真机(yizhi-robot/src_remote/ROBOT_API.md)有这些关键能力:
  /left/fk                    正运动学:关节角 → 末端位姿   ← 自标定的关键
  /left/ik                    逆运动学
  /left/get_current_tool_frame 工具坐标系偏移(TCP)          ← 一直缺的数据
  /left/get_work_frame        工作坐标系
  /left/joint_states          关节状态

若仿真也有 fk,自标定就不必靠视觉找手:移动 → fk 得精确 base 位姿 → 同时拍照,
即可造出任意多个 Z 分散的标定点,替掉 gt.json 那 4 个共面点。

本脚本扫三层:SDK 类方法签名 / ROS2 topic+service / SDK 底层通信对象。

用法:  python -m tools.scan_sdk
"""

import inspect
import logging

logging.basicConfig(level=logging.WARNING)


def show(obj, name, depth=0):
    pad = "  " * depth
    for m in sorted(x for x in dir(obj) if not x.startswith("_")):
        try:
            a = getattr(obj, m)
        except Exception:
            continue
        if callable(a):
            try:
                print("%s  %s%s" % (pad, m, inspect.signature(a)))
            except (ValueError, TypeError):
                print("%s  %s(...)" % (pad, m))
        else:
            v = repr(a)
            print("%s  %s = %s" % (pad, m, v[:90]))


def main():
    import rabo_robocap as R
    from agents.nut_picker import config as C

    print("=" * 72)
    print("[1] rabo_robocap 顶层导出")
    print("=" * 72)
    print("  " + ", ".join(x for x in dir(R) if not x.startswith("_")))

    print("\n" + "=" * 72)
    print("[2] LinkerArmA7 类方法签名(对照真机 fk/ik/tool_frame)")
    print("=" * 72)
    show(R.LinkerArmA7, "LinkerArmA7")

    print("\n" + "=" * 72)
    print("[3] ArmBase / ArmClient(底层可能暴露更多)")
    print("=" * 72)
    for cls in ("ArmBase", "ArmClient"):
        if hasattr(R, cls):
            print("\n  ── %s ──" % cls)
            show(getattr(R, cls), cls)

    print("\n" + "=" * 72)
    print("[4] LinkerHandO6Left(手的接口 / TCP 偏移)")
    print("=" * 72)
    show(R.LinkerHandO6Left, "LinkerHandO6Left")

    print("\n" + "=" * 72)
    print("[5] 实例化后的属性(连接对象 / 内部状态)")
    print("=" * 72)
    try:
        arm = R.LinkerArmA7(robot_id=C.LEFT_ARM_ID, mode="sim")
        for k, v in vars(arm).items():
            print("  %s = %s" % (k, repr(v)[:100]))
        print("\n  尝试 fk / ik / tool_frame 类方法:")
        for m in ("fk", "ik", "forward_kinematics", "inverse_kinematics",
                  "get_tool_frame", "get_current_tool_frame", "get_work_frame",
                  "get_joint_limit", "get_joint_angles"):
            if hasattr(arm, m):
                try:
                    print("    %s() → %r" % (m, getattr(arm, m)()))
                except Exception as e:
                    print("    %s() 需参数/异常: %s" % (m, str(e)[:70]))
        try:
            arm.shutdown()
        except Exception:
            pass
    except Exception as e:
        print("  实例化失败(需仿真运行): %s" % str(e)[:120])

    print("\n" + "=" * 72)
    print("[6] ROS2 topic / service 全表")
    print("=" * 72)
    try:
        import rclpy
        from rclpy.node import Node
        if not rclpy.ok():
            rclpy.init()
        n = Node("sdk_scan")
        import time
        time.sleep(2.0)
        tops = n.get_topic_names_and_types()
        srvs = n.get_service_names_and_types()
        print("\n  ── %d topics ──" % len(tops))
        for t, ty in sorted(tops):
            print("    %-58s %s" % (t, ",".join(ty)))
        print("\n  ── %d services ──" % len(srvs))
        for s, ty in sorted(srvs):
            print("    %-58s %s" % (s, ",".join(ty)))
        # 重点找 fk / ik / camera_info / tf
        print("\n  ── 关键接口筛查 ──")
        allnames = [t for t, _ in tops] + [s for s, _ in srvs]
        for kw in ("fk", "ik", "camera_info", "tf", "tool", "frame",
                   "joint", "描述", "description"):
            hit = [a for a in allnames if kw in a.lower()]
            print("    %-14s: %s" % (kw, hit[:6] if hit else "无"))
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as e:
        print("  ROS2 扫描失败: %s" % str(e)[:150])


if __name__ == "__main__":
    main()
