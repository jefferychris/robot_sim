"""查 TF 树,拿 base_link → 头部相机 的解析变换,替掉 4 点硬拟合。

为什么现有变换是错的:
  HEAD_TO_BASE_T 是拿 3 螺母 + 1 箱子的坐标真值 Kabsch 反解出来的。4 点几乎
  共面(Z 差 < 2cm),旋转分量极不稳定;且 GT 用的是**世界坐标**,解出来的其实
  是 相机→世界,却当 相机→base 在用。误差 43.7mm 且换位置就失准。

场景 JSON 只给了链条的一半:
  相机 → obzg_33 : xyz=[0.05,0,-0.03] rpy=[0,0.8,0]   ← 有
  obzg_33 → base_link                                   ← 缺
平台是 ROS2,这段关系在 TF 树里,查出来就是场景定义的真值,无需拟合。

用法:  python -m tools.query_tf
"""

import math
import time

import numpy as np


def mat_from_tf(t):
    """geometry_msgs/TransformStamped → 4x4"""
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z]
    return T


def rpy_of(R):
    return (math.atan2(R[2, 1], R[2, 2]),
            math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])),
            math.atan2(R[1, 0], R[0, 0]))


def main():
    import rclpy
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener

    if not rclpy.ok():
        rclpy.init()
    node = Node("tf_probe")
    buf = Buffer()
    TransformListener(buf, node)

    print("收集 TF (5s)...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)

    # ── 1. 列出所有 frame ────────────────────────────────────────
    raw = buf.all_frames_as_string()
    print("\n" + "=" * 72)
    print("[1] TF 树")
    print("=" * 72)
    print(raw if raw.strip() else "  (空 —— 该平台可能不发布 TF)")

    frames = sorted(set(
        w.strip("'\"") for line in raw.splitlines()
        for w in line.replace("Frame ", "").replace(" exists with parent ", " ").split()
        if w and not w.endswith(".")
    ))
    print("\n  解析出 %d 个 frame" % len(frames))

    # ── 2. 找 base 和 camera 候选 ────────────────────────────────
    bases = [f for f in frames if "base" in f.lower() or "link_5" in f.lower()]
    cams = [f for f in frames if any(k in f.lower()
            for k in ("cam", "sensor", "obzg", "dcam"))]
    print("\n[2] 候选")
    print("  base 类 : %s" % (bases[:8] or "无"))
    print("  camera 类: %s" % (cams[:12] or "无"))

    # ── 3. 逐对查变换 ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("[3] base → camera 变换")
    print("=" * 72)
    found = []
    for b in bases[:6]:
        for c in cams[:12]:
            try:
                tf = buf.lookup_transform(b, c, rclpy.time.Time())
            except Exception:
                continue
            T = mat_from_tf(tf)
            r, p, yw = rpy_of(T[:3, :3])
            print("\n  %s → %s" % (b, c))
            print("    xyz = (%.4f, %.4f, %.4f)" % tuple(T[:3, 3]))
            print("    rpy = (%.4f, %.4f, %.4f) rad = (%.1f, %.1f, %.1f)°"
                  % (r, p, yw, math.degrees(r), math.degrees(p), math.degrees(yw)))
            found.append((b, c, T))

    if not found:
        print("\n  未查到 —— TF 可能未发布,走兜底方案(见下)")
    else:
        print("\n" + "=" * 72)
        print("[4] 可直接贴进 config.py 的 HEAD_TO_BASE_T")
        print("=" * 72)
        for b, c, T in found:
            if "dcam" in c.lower() or "depth" in c.lower():
                print("\n# %s → %s" % (b, c))
                print("HEAD_TO_BASE_T = np.array([")
                for row in T:
                    print("    [%s]," % ", ".join("% .8f" % v for v in row))
                print("], dtype=np.float64)")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
