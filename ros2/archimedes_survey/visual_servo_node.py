"""
visual_servo_node.py
====================
ROS2 視覺伺服節點 (Visual Servo Node)
阿基米德螺旋探勘船 — 洞穴中心點→底座精準定位

功能：
  訂閱 YOLO 洞穴偵測結果，使用比例控制器驅動底座移動，
  直到最近洞穴中心距離 < 0.35m（ARM_REACH_M），
  此時發布對準完成信號，讓 auto_navigate 切入 SETTLING→SCAN 模式。

Subscribe:
  /camera/burrow_detections  (std_msgs/String -- JSON 陣列)
    每筆 detection 格式：
      {
        "center_world_m": [dx, dy],  # 相機座標系下相對距離（公尺）
        "distance_m": 0.72,          # 到洞穴中心的直線距離
        "confidence": 0.91,
        "burrow_id": "b001"
      }

Publish:
  /cmd_vel                   (geometry_msgs/Twist) -- 微調速度指令
  /visual_servo/aligned      (std_msgs/Bool)        -- 對準完成旗標
  /visual_servo/status       (std_msgs/String -- JSON) -- 除錯狀態

Parameters:
  arm_reach_m    (float, default 0.35)  -- 觸發對準的距離閾值
  max_approach_m (float, default 1.5)   -- 超過此距離忽略（太遠）
  kp_linear      (float, default 0.3)   -- 比例增益（線速度）
  kp_angular     (float, default 0.8)   -- 比例增益（角速度）
  max_v          (float, default 0.1)   -- 最大線速度 m/s
  max_w          (float, default 0.2)   -- 最大角速度 rad/s
  no_detect_timeout_s (float, default 5.0) -- 無偵測多久後 aligned=False

Run:
  ros2 run archimedes_survey visual_servo_node

設計說明（IBVS 簡化版）：
  - dx = center_world_m[0]：橫向誤差（正=右，負=左）
  - dy = center_world_m[1]：縱向誤差（正=前，負=後）
  - 線速度 v = Kp * (dist - ARM_REACH_M)，帶最大值截斷
  - 角速度 w = -Kp_w * dx（負號：目標偏右→左轉）
  - 距離 < ARM_REACH_M：停止輸出、aligned=True
  - 距離 > MAX_APPROACH_M：目標太遠，不輸出
  - 超過 no_detect_timeout_s 無偵測：aligned=False，停止
"""

import json
import math
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, String
except ImportError:
    sys.exit("ROS2 not found. Source /opt/ros/<distro>/setup.bash first.")


# ---------------------------------------------------------------------------
# 預設參數（可由 ROS2 params 覆蓋）
# ---------------------------------------------------------------------------
DEFAULT_ARM_REACH_M       = 0.35   # 機械臂最大有效臂展（已修正 from 1.0m）
DEFAULT_MAX_APPROACH_M    = 1.5    # 超過此距離不伺服
DEFAULT_KP_LINEAR         = 0.3    # 線速度比例增益（PI 控制器）
DEFAULT_KP_ANGULAR        = 0.8    # 角速度比例增益
DEFAULT_KI_LINEAR         = 0.05   # 線速度積分增益（消除沙地摩擦穩態誤差）
DEFAULT_KI_ANGULAR        = 0.10   # 角速度積分增益
DEFAULT_MAX_V             = 0.10   # 最大線速度 m/s（不超過模擬值 0.088）
DEFAULT_MAX_W             = 0.20   # 最大角速度 rad/s
DEFAULT_NO_DETECT_TIMEOUT = 5.0    # 無偵測超時 (s)
DEFAULT_MIN_CONFIDENCE    = 0.6    # 最低信心閾值（過濾反光/陰影誤判）
DEFAULT_LOCK_HYSTERESIS_M = 0.15   # 洞穴鎖定遲滯距離（防止多洞穴間擺盪）
CONTROL_HZ                = 10.0   # 控制迴路頻率
INTEGRAL_WINDUP_MAX       = 0.30   # 積分防飽和上限（避免快速切換時積累過大）


class VisualServoNode(Node):
    """視覺伺服節點：洞穴中心→底座精準對位"""

    def __init__(self):
        super().__init__("visual_servo_node")

        # --- 宣告 ROS2 參數 ---
        self.declare_parameter("arm_reach_m",        DEFAULT_ARM_REACH_M)
        self.declare_parameter("max_approach_m",     DEFAULT_MAX_APPROACH_M)
        self.declare_parameter("kp_linear",          DEFAULT_KP_LINEAR)
        self.declare_parameter("kp_angular",         DEFAULT_KP_ANGULAR)
        self.declare_parameter("ki_linear",          DEFAULT_KI_LINEAR)
        self.declare_parameter("ki_angular",         DEFAULT_KI_ANGULAR)
        self.declare_parameter("max_v",              DEFAULT_MAX_V)
        self.declare_parameter("max_w",              DEFAULT_MAX_W)
        self.declare_parameter("no_detect_timeout_s",  DEFAULT_NO_DETECT_TIMEOUT)
        self.declare_parameter("min_confidence",       DEFAULT_MIN_CONFIDENCE)
        self.declare_parameter("lock_hysteresis_m",    DEFAULT_LOCK_HYSTERESIS_M)

        # --- 讀取參數 ---
        self._arm_reach       = self.get_parameter("arm_reach_m").value
        self._max_approach    = self.get_parameter("max_approach_m").value
        self._kp_lin          = self.get_parameter("kp_linear").value
        self._kp_ang          = self.get_parameter("kp_angular").value
        self._ki_lin          = self.get_parameter("ki_linear").value
        self._ki_ang          = self.get_parameter("ki_angular").value
        self._max_v           = self.get_parameter("max_v").value
        self._max_w           = self.get_parameter("max_w").value
        self._timeout         = self.get_parameter("no_detect_timeout_s").value
        self._min_conf        = self.get_parameter("min_confidence").value
        self._lock_hyst       = self.get_parameter("lock_hysteresis_m").value

        # --- 狀態變數 ---
        self._last_detect_time: float = 0.0    # 上次收到偵測的時間戳
        self._aligned: bool = False             # 是否已對準
        self._latest_detections: list = []      # 最新偵測結果列表
        self._servo_active: bool = False        # 伺服是否啟動中
        self._locked_burrow_id: str | None = None  # 當前鎖定洞穴 ID（防擺盪）
        # PI 積分項（消除沙地摩擦穩態誤差）
        self._integral_dist: float    = 0.0
        self._integral_heading: float = 0.0
        self._last_ctrl_t: float      = time.monotonic()

        # --- 訂閱者 ---
        self._sub_detections = self.create_subscription(
            String,
            "/camera/burrow_detections",
            self._cb_detections,
            10
        )

        # --- 發布者 ---
        self._pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self._pub_aligned = self.create_publisher(Bool, "/visual_servo/aligned", 10)
        self._pub_status  = self.create_publisher(String, "/visual_servo/status", 10)

        # --- 控制迴路計時器 ---
        self._timer = self.create_timer(
            1.0 / CONTROL_HZ,
            self._control_loop
        )

        self.get_logger().info(
            f"[VisualServo] 啟動（PI控制器）：arm_reach={self._arm_reach}m, "
            f"kp_lin={self._kp_lin}, ki_lin={self._ki_lin}, "
            f"kp_ang={self._kp_ang}, ki_ang={self._ki_ang}, "
            f"max_v={self._max_v}m/s, timeout={self._timeout}s"
        )

    # -----------------------------------------------------------------------
    # 訂閱回呼：解析 JSON 偵測陣列
    # -----------------------------------------------------------------------
    def _cb_detections(self, msg: String):
        """接收洞穴偵測 JSON，更新最新偵測快取"""
        try:
            data = json.loads(msg.data)
            # 支援陣列格式 [...] 或單筆 {...} 兩種
            if isinstance(data, dict):
                data = [data]
            self._latest_detections = data
            self._last_detect_time = time.monotonic()
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"[VisualServo] JSON 解析失敗: {e}")

    # -----------------------------------------------------------------------
    # 控制迴路：每 1/CONTROL_HZ 秒執行
    # -----------------------------------------------------------------------
    def _control_loop(self):
        now = time.monotonic()
        elapsed_since_detect = now - self._last_detect_time

        # --- 情況 1：超時無偵測 ---
        if elapsed_since_detect > self._timeout:
            if self._aligned or self._servo_active:
                self.get_logger().info(
                    f"[VisualServo] 超過 {self._timeout:.0f}s 無偵測 → aligned=False"
                )
            self._aligned = False
            self._servo_active = False
            self._integral_dist = 0.0        # 重置積分（防止殘留積分影響下次伺服）
            self._integral_heading = 0.0
            self._publish_stop()
            self._publish_status(
                state="timeout",
                message=f"無偵測 {elapsed_since_detect:.1f}s"
            )
            return

        # --- 尋找最近的洞穴 ---
        target = self._select_nearest_burrow()

        # --- 情況 2：無有效目標（全部超出範圍或列表空）---
        if target is None:
            self._publish_stop()
            self._publish_status(state="no_valid_target", message="無有效洞穴目標")
            return

        dist   = target["distance_m"]
        dx     = target["center_world_m"][0]   # 橫向誤差（正=右）
        dy     = target["center_world_m"][1]   # 縱向誤差（正=前）
        bid    = target.get("burrow_id", "?")

        # --- 情況 3：已在 ARM_REACH 範圍內 → 對準完成 ---
        if dist <= self._arm_reach:
            if not self._aligned:
                self.get_logger().info(
                    f"[VisualServo] ✓ 對準完成！dist={dist:.3f}m ≤ {self._arm_reach}m "
                    f"(burrow={bid})"
                )
            self._aligned = True
            self._servo_active = False
            self._integral_dist = 0.0        # 對準後重置積分
            self._integral_heading = 0.0
            self._publish_stop()
            self._publish_aligned(True)
            self._publish_status(
                state="aligned",
                dist_m=dist,
                burrow_id=bid,
                message="對準完成，等待 SETTLING→SCAN"
            )
            return

        # --- 情況 4：目標太遠（> max_approach_m）→ 不伺服 ---
        if dist > self._max_approach:
            self._integral_dist = 0.0        # 重置積分（目標超出範圍）
            self._integral_heading = 0.0
            self._publish_stop()
            self._publish_aligned(False)
            self._publish_status(
                state="too_far",
                dist_m=dist,
                burrow_id=bid,
                message=f"目標過遠 {dist:.2f}m > {self._max_approach}m"
            )
            return

        # --- 情況 5：在伺服範圍（arm_reach < dist <= max_approach）→ PI 控制 ---
        self._aligned = False
        self._servo_active = True

        # 計算 dt（用於積分）
        t_now = time.monotonic()
        dt = min(t_now - self._last_ctrl_t, 0.5)  # 限制 dt 上限，防止長暫停後積分爆衝
        self._last_ctrl_t = t_now

        # 方位角誤差（rad，正=目標偏右）
        heading_err = math.atan2(dx, dy)

        # 線速度誤差 = dist - arm_reach
        dist_err = dist - self._arm_reach

        # 積分累積（含防飽和 anti-windup）
        self._integral_dist    = max(-INTEGRAL_WINDUP_MAX,
                                     min(INTEGRAL_WINDUP_MAX,
                                         self._integral_dist + dist_err * dt))
        self._integral_heading = max(-INTEGRAL_WINDUP_MAX,
                                     min(INTEGRAL_WINDUP_MAX,
                                         self._integral_heading + heading_err * dt))

        # PI 控制輸出
        v_raw = self._kp_lin * dist_err + self._ki_lin * self._integral_dist
        w_raw = -(self._kp_ang * heading_err + self._ki_ang * self._integral_heading)

        v = max(0.0, min(v_raw, self._max_v))               # 只允許前進
        w = max(-self._max_w, min(self._max_w, w_raw))

        self._publish_cmd_vel(v, w)
        self._publish_aligned(False)
        self._publish_status(
            state="servoing",
            dist_m=dist,
            dx_m=dx,
            dy_m=dy,
            v_cmd=round(v, 4),
            w_cmd=round(w, 4),
            integral_dist=round(self._integral_dist, 4),
            integral_heading=round(self._integral_heading, 4),
            burrow_id=bid,
            message=f"PI 伺服中 dist={dist:.3f}m heading_err={math.degrees(heading_err):.1f}°"
        )

        self.get_logger().debug(
            f"[VisualServo] PI: dist={dist:.3f}m v={v:.4f}m/s w={w:.4f}rad/s "
            f"I_dist={self._integral_dist:.3f} I_hdg={self._integral_heading:.3f} bid={bid}"
        )

    # -----------------------------------------------------------------------
    # 輔助方法
    # -----------------------------------------------------------------------
    def _select_nearest_burrow(self):
        """
        從偵測列表中選出最佳目標洞穴。

        策略（防止多洞穴間擺盪）：
        1. 過濾低信心和超出範圍的偵測
        2. 若有已鎖定的洞穴 ID，優先繼續追蹤（除非它已進入 aligned 範圍）
        3. 若鎖定洞穴消失或超出範圍，切換至最近洞穴並更新鎖定

        lock_hysteresis_m：切換目標需新目標比當前目標近 hyst 公尺以上，
        避免在距離相近的洞穴間頻繁切換。
        """
        valid = []
        for det in self._latest_detections:
            try:
                dist = float(det["distance_m"])
                conf = float(det.get("confidence", 1.0))
                cw   = det["center_world_m"]
                if len(cw) < 2:
                    continue
                if dist > self._max_approach:
                    continue
                # 過濾低信心偵測（防止反光/陰影誤判）
                if conf < self._min_conf:
                    continue
                valid.append({**det, "distance_m": dist})
            except (KeyError, TypeError, ValueError):
                continue

        if not valid:
            self._locked_burrow_id = None
            return None

        # 嘗試維持當前鎖定
        if self._locked_burrow_id is not None:
            locked = next(
                (d for d in valid if d.get("burrow_id") == self._locked_burrow_id),
                None
            )
            if locked is not None:
                # 鎖定有效：檢查是否有遠遠更近的目標（超過 hysteresis）
                nearest = min(valid, key=lambda d: d["distance_m"])
                if nearest["distance_m"] < locked["distance_m"] - self._lock_hyst:
                    # 切換至更近目標
                    self._locked_burrow_id = nearest.get("burrow_id")
                    self.get_logger().info(
                        f"[VisualServo] 切換鎖定洞穴：{self._locked_burrow_id} "
                        f"（距離差 > {self._lock_hyst}m）"
                    )
                    return nearest
                return locked

        # 無鎖定或鎖定失效：選最近
        nearest = min(valid, key=lambda d: d["distance_m"])
        self._locked_burrow_id = nearest.get("burrow_id")
        return nearest

    def _publish_cmd_vel(self, v: float, w: float):
        """發布速度指令"""
        twist = Twist()
        twist.linear.x  = float(v)
        twist.angular.z = float(w)
        self._pub_cmd_vel.publish(twist)

    def _publish_stop(self):
        """發布零速度（停止）"""
        self._pub_cmd_vel.publish(Twist())

    def _publish_aligned(self, aligned: bool):
        """發布對準旗標"""
        msg = Bool()
        msg.data = aligned
        self._pub_aligned.publish(msg)

    def _publish_status(self, **kwargs):
        """發布 JSON 狀態字串"""
        payload = {"timestamp": time.time(), **kwargs}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub_status.publish(msg)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[VisualServo] 手動中斷，關閉節點")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
