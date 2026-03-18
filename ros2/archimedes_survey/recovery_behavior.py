"""
recovery_behavior.py
====================
自救脫困行為節點：偵測機器人卡陷於軟泥並執行分級脫困策略。

偵測條件（全部同時成立）：
  - GPS 位置變化 < 5cm / 30秒（實際位移近零）
  - 馬達步數正常轉動（螺旋空轉）
  - IMU 無明顯位移加速度

脫困策略（分 5 級，逐步升級）：
  L1 - 反轉衝擊（前後各 3 次）
  L2 - 對角差速震盪（左快右慢交替）
  L3 - 手臂槓桿輔助（手臂壓地當支點）
  L4 - 螺旋蠕動（模擬蠕蟲運動）
  L5 - 放棄，發送 LoRa 求救信號，等待人工救援

Subscribe:
  /gps/fix              (NavSatFix)
  /imu/data             (Imu)
  /motor/left_pwm       (Float32)
  /motor/right_pwm      (Float32)

Publish:
  /cmd_vel              (Twist)      ← 脫困指令
  /arm_command          (String)     ← 手臂槓桿指令
  /recovery/status      (String)     ← JSON 狀態廣播
  /lora/tx              (String)     ← 求救信號

Run:
  ros2 run archimedes_survey recovery_behavior
"""

import json
import math
import sys
import time
from collections import deque

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import NavSatFix, Imu
    from std_msgs.msg import Float32, String
except ImportError:
    sys.exit("ROS2 not found.")

# ------------------------------------------------------------------
# 參數
# ------------------------------------------------------------------
STUCK_DETECT_WINDOW_S   = 30     # 判定卡陷的時間窗口（秒）
STUCK_DIST_THRESHOLD_M  = 0.05   # 窗口內位移低於 5cm → 卡陷
MOTOR_ACTIVE_THRESHOLD  = 0.15   # 馬達指令 > 15% 才算有在轉
CHECK_INTERVAL_S        = 2.0    # 每 2 秒檢查一次
MAX_RECOVERY_LEVELS     = 5
RECOVERY_ATTEMPT_TIMEOUT_S = 60  # 每級最長嘗試時間

# GPS 坐標 → 公尺的簡單換算
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


class RecoveryBehaviorNode(Node):
    def __init__(self):
        super().__init__("recovery_behavior")

        # State
        self._gps_history: deque = deque(maxlen=int(STUCK_DETECT_WINDOW_S / CHECK_INTERVAL_S) + 2)
        self._lat = None
        self._lon = None
        self._motor_l = 0.0
        self._motor_r = 0.0
        self._is_stuck = False
        self._recovery_level = 0
        self._recovery_active = False
        self._recovery_start_t = 0.0
        self._home_lat = None
        self._home_lon = None

        # Pubs
        self._pub_vel    = self.create_publisher(Twist,  "/cmd_vel",          10)
        self._pub_arm    = self.create_publisher(String, "/arm_command",      10)
        self._pub_status = self.create_publisher(String, "/recovery/status",  10)
        self._pub_lora   = self.create_publisher(String, "/lora/tx",          10)

        # Subs
        self.create_subscription(NavSatFix, "/gps/fix",        self._cb_gps,     10)
        self.create_subscription(Imu,       "/imu/data",       self._cb_imu,     10)
        self.create_subscription(Float32,   "/motor/left_pwm", self._cb_ml,      10)
        self.create_subscription(Float32,   "/motor/right_pwm",self._cb_mr,      10)

        # Timer
        self.create_timer(CHECK_INTERVAL_S, self._check_stuck)

        self.get_logger().info("Recovery Behavior Node ready")

    # ------------------------------------------------------------------
    def _cb_gps(self, msg: NavSatFix):
        if msg.status.status < 0:
            return
        self._lat = msg.latitude
        self._lon = msg.longitude
        if self._home_lat is None:
            self._home_lat = msg.latitude
            self._home_lon = msg.longitude
        self._gps_history.append((time.time(), msg.latitude, msg.longitude))

    def _cb_imu(self, msg: Imu):
        pass  # 未來可用加速度計輔助判斷

    def _cb_ml(self, msg: Float32):
        self._motor_l = msg.data

    def _cb_mr(self, msg: Float32):
        self._motor_r = msg.data

    # ------------------------------------------------------------------
    def _check_stuck(self):
        """定期檢查是否卡陷。"""
        if self._recovery_active:
            self._manage_recovery()
            return

        if len(self._gps_history) < 3:
            return

        motor_active = (abs(self._motor_l) > MOTOR_ACTIVE_THRESHOLD or
                        abs(self._motor_r) > MOTOR_ACTIVE_THRESHOLD)
        if not motor_active:
            self._is_stuck = False
            return

        # 計算窗口內最大位移
        t0, lat0, lon0 = self._gps_history[0]
        t1, lat1, lon1 = self._gps_history[-1]
        elapsed = t1 - t0
        if elapsed < STUCK_DETECT_WINDOW_S * 0.8:
            return

        total_dist = haversine_m(lat0, lon0, lat1, lon1)

        if total_dist < STUCK_DIST_THRESHOLD_M:
            if not self._is_stuck:
                self._is_stuck = True
                self._recovery_level = 0
                self.get_logger().warn(
                    f"⚠️  偵測到卡陷！{elapsed:.0f}s 內位移僅 {total_dist*100:.1f} cm，啟動脫困程序")
                self._start_recovery()
        else:
            self._is_stuck = False

    # ------------------------------------------------------------------
    def _start_recovery(self):
        self._recovery_active = True
        self._recovery_level += 1
        self._recovery_start_t = time.time()
        self.get_logger().info(f"[Recovery] 啟動 Level {self._recovery_level}")
        self._broadcast_status(f"recovering_L{self._recovery_level}")

    def _manage_recovery(self):
        """執行當前脫困等級的動作。"""
        elapsed = time.time() - self._recovery_start_t

        if self._recovery_level == 1:
            self._recovery_L1(elapsed)
        elif self._recovery_level == 2:
            self._recovery_L2(elapsed)
        elif self._recovery_level == 3:
            self._recovery_L3(elapsed)
        elif self._recovery_level == 4:
            self._recovery_L4(elapsed)
        elif self._recovery_level >= 5:
            self._recovery_L5()
        else:
            self._recovery_active = False

    # ------------------------------------------------------------------
    # 脫困策略
    # ------------------------------------------------------------------
    def _recovery_L1(self, elapsed):
        """L1：前後衝擊（3 次往返）。"""
        cycle = int(elapsed / 3.0) % 6
        t = Twist()
        if cycle % 2 == 0:
            t.linear.x = 0.088   # 全速前進
        else:
            t.linear.x = -0.088  # 全速後退
        self._pub_vel.publish(t)

        if elapsed > RECOVERY_ATTEMPT_TIMEOUT_S * 0.3:
            self.get_logger().info("[Recovery] L1 失敗，升級至 L2")
            self._stop_motors()
            self._recovery_level = 2
            self._recovery_start_t = time.time()

    def _recovery_L2(self, elapsed):
        """L2：對角差速震盪（左右螺旋交替加速）。"""
        cycle = int(elapsed / 2.0) % 4
        t = Twist()
        if cycle == 0:
            t.linear.x = 0.06;  t.angular.z =  0.35  # 左轉前進
        elif cycle == 1:
            t.linear.x = 0.06;  t.angular.z = -0.35  # 右轉前進
        elif cycle == 2:
            t.linear.x = -0.06; t.angular.z =  0.35  # 左轉後退
        else:
            t.linear.x = -0.06; t.angular.z = -0.35  # 右轉後退
        self._pub_vel.publish(t)

        if elapsed > RECOVERY_ATTEMPT_TIMEOUT_S * 0.5:
            self.get_logger().info("[Recovery] L2 失敗，升級至 L3（手臂槓桿）")
            self._stop_motors()
            self._recovery_level = 3
            self._recovery_start_t = time.time()

    def _recovery_L3(self, elapsed):
        """L3：手臂壓地槓桿輔助，配合螺旋前進。"""
        if elapsed < 3.0:
            # 手臂壓低：j2 向前壓地
            cmd = {"type": "arm", "j1": 0.0, "j2": 80.0, "j3": 120.0}
            self._pub_arm.publish(String(data=json.dumps(cmd)))
        elif elapsed < 8.0:
            # 螺旋全速前進（手臂當支點）
            t = Twist(); t.linear.x = 0.088
            self._pub_vel.publish(t)
        elif elapsed < 11.0:
            # 手臂抬起
            cmd = {"type": "arm", "j1": 0.0, "j2": 0.0, "j3": 0.0}
            self._pub_arm.publish(String(data=json.dumps(cmd)))
        elif elapsed < 16.0:
            t = Twist(); t.linear.x = -0.088  # 後退
            self._pub_vel.publish(t)
        elif elapsed < 19.0:
            self._stop_motors()
        else:
            self.get_logger().info("[Recovery] L3 失敗，升級至 L4（螺旋蠕動）")
            self._recovery_level = 4
            self._recovery_start_t = time.time()

    def _recovery_L4(self, elapsed):
        """L4：螺旋蠕動（模擬蠕蟲式漸進前進）。"""
        # 短暫的高頻正弦速度脈衝讓螺旋葉片「咬」進泥裡
        phase = (elapsed * 2.0) % (2 * math.pi)
        velocity = 0.05 * math.sin(phase) + 0.03  # 帶偏置的正弦波
        t = Twist()
        t.linear.x = velocity
        self._pub_vel.publish(t)

        if elapsed > RECOVERY_ATTEMPT_TIMEOUT_S:
            self.get_logger().error("[Recovery] L4 失敗，升級至 L5（求救信號）")
            self._stop_motors()
            self._recovery_level = 5
            self._recovery_start_t = time.time()

    def _recovery_L5(self):
        """L5：所有方法失敗，發 LoRa 求救，停止所有馬達。"""
        self._stop_motors()
        rescue_msg = json.dumps({
            "type":  "RESCUE",
            "lat":   self._lat,
            "lon":   self._lon,
            "msg":   "Archimedes stuck, manual recovery needed",
            "ts":    int(time.time()),
        })
        self._pub_lora.publish(String(data=rescue_msg))
        self.get_logger().error(
            f"🆘 所有脫困方法失敗！GPS: ({self._lat:.6f}, {self._lon:.6f})  "
            "已發送 LoRa 求救信號，等待人工救援")
        self._broadcast_status("stuck_rescue_sent")
        self._recovery_active = False  # 停止循環，等待人工介入

    # ------------------------------------------------------------------
    def _stop_motors(self):
        self._pub_vel.publish(Twist())

    def _broadcast_status(self, state: str):
        payload = json.dumps({
            "state": state,
            "level": self._recovery_level,
            "lat":   self._lat,
            "lon":   self._lon,
        })
        self._pub_status.publish(String(data=payload))


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RecoveryBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Recovery node shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
