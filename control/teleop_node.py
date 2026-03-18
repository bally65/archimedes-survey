"""
teleop_node.py
==============
ROS2 node: collects all sensor data into a single /telemetry JSON topic,
and translates /cmd_vel into left/right motor PWM outputs.

Subscribe:
  /gps/fix         (sensor_msgs/NavSatFix)
  /imu/data        (sensor_msgs/Imu)
  /tidal/range_cm  (std_msgs/Float32)
  /battery/voltage (std_msgs/Float32)
  /arm/joint_states (sensor_msgs/JointState)
  /burrow_detections (std_msgs/String -- JSON from vision node)

Publish:
  /telemetry       (std_msgs/String -- JSON to web dashboard)
  /motor/left_pwm  (std_msgs/Float32  -1.0 to +1.0)
  /motor/right_pwm (std_msgs/Float32  -1.0 to +1.0)

Run:
    ros2 run archimedes_survey teleop_node
"""

import json
import math
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import NavSatFix, Imu, JointState
    from std_msgs.msg import Float32, String, Bool
except ImportError:
    sys.exit("ROS2 not found. Source: source /opt/ros/<distro>/setup.bash")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WHEEL_BASE_M   = 0.30      # lateral separation of two screws [m]
MAX_RPM        = 60.0      # NEMA23 + 10:1 gearbox at full speed
BATTERY_FULL_V = 12.6      # 3S LiPo
BATTERY_EMPTY_V = 10.5

TIDAL_WARN_CM  = 10.0      # trigger tidal alert above this water level
TELEMETRY_HZ   = 5.0       # publish rate


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")

        # --- shared state ---
        self._gps     = {"lat": 0.0, "lon": 0.0, "fix": False}
        self._imu     = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self._battery = {"voltage": BATTERY_FULL_V, "pct": 100}
        self._tidal   = {"water_cm": 0.0, "alert": False}
        self._motors  = {"left_rpm": 0.0, "right_rpm": 0.0}
        self._arm     = {"j1_deg": 0.0, "j2_deg": 0.0, "j3_deg": 0.0, "pose": "stow"}
        self._lora    = {"connected": False, "rssi": -999}
        self._auto    = {"enabled": False, "status": "idle",
                         "target_lat": 0.0, "target_lon": 0.0}
        self._last_cmd_vel_t = 0.0

        # --- publishers ---
        self._pub_tel  = self.create_publisher(String,  "/telemetry",       10)
        self._pub_lpwm = self.create_publisher(Float32, "/motor/left_pwm",  10)
        self._pub_rpwm = self.create_publisher(Float32, "/motor/right_pwm", 10)

        # --- subscribers ---
        self.create_subscription(NavSatFix, "/gps/fix",          self._cb_gps,     10)
        self.create_subscription(Imu,       "/imu/data",         self._cb_imu,     10)
        self.create_subscription(Float32,   "/tidal/range_cm",   self._cb_tidal,   10)
        self.create_subscription(Float32,   "/battery/voltage",  self._cb_battery, 10)
        self.create_subscription(JointState,"/arm/joint_states", self._cb_arm,     10)
        self.create_subscription(Twist,     "/cmd_vel",          self._cb_cmdvel,  10)
        self.create_subscription(String,    "/lora/status",      self._cb_lora,    10)
        self.create_subscription(String,    "/auto_command",     self._cb_auto,    10)

        # --- telemetry timer ---
        self.create_timer(1.0 / TELEMETRY_HZ, self._publish_telemetry)

        # --- watchdog: stop motors if no cmd_vel for 1 s ---
        self.create_timer(0.5, self._watchdog)

        self.get_logger().info("TeleopNode started")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _cb_gps(self, msg: NavSatFix):
        self._gps = {
            "lat": msg.latitude,
            "lon": msg.longitude,
            "fix": msg.status.status >= 0,
        }

    def _cb_imu(self, msg: Imu):
        q = msg.orientation
        # quaternion → Euler (ZYX)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll  = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw   = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        self._imu = {"roll": round(roll, 1), "pitch": round(pitch, 1),
                     "yaw": round(yaw, 1)}

    def _cb_tidal(self, msg: Float32):
        cm = msg.data
        self._tidal = {
            "water_cm": round(cm, 1),
            "alert":    cm >= TIDAL_WARN_CM,
        }
        if self._tidal["alert"]:
            self.get_logger().warn(f"TIDAL ALERT: {cm:.1f} cm")

    def _cb_battery(self, msg: Float32):
        v   = msg.data
        pct = (v - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V) * 100
        pct = max(0.0, min(100.0, pct))
        self._battery = {"voltage": round(v, 2), "pct": round(pct, 1)}
        if pct < 20:
            self.get_logger().warn(f"LOW BATTERY: {pct:.0f}%")

    def _cb_arm(self, msg: JointState):
        names = {n: math.degrees(msg.position[i])
                 for i, n in enumerate(msg.name)
                 if i < len(msg.position)}
        self._arm.update({
            "j1_deg": round(names.get("j1_yaw", 0), 1),
            "j2_deg": round(names.get("j2_shoulder", 0), 1),
            "j3_deg": round(names.get("j3_elbow", 0), 1),
        })

    def _cb_cmdvel(self, msg: Twist):
        self._last_cmd_vel_t = time.monotonic()
        v  = msg.linear.x   # m/s  forward
        w  = msg.angular.z  # rad/s turn

        # differential drive:  v_L = v - w*d/2,  v_R = v + w*d/2
        v_l = v - w * WHEEL_BASE_M / 2.0
        v_r = v + w * WHEEL_BASE_M / 2.0

        max_v = MAX_RPM / 60.0 * 0.1  # rough m/s at max RPM (adjust with actual gear)
        pwm_l = max(-1.0, min(1.0, v_l / max_v))
        pwm_r = max(-1.0, min(1.0, v_r / max_v))

        self._motors = {
            "left_rpm":  round(pwm_l * MAX_RPM, 1),
            "right_rpm": round(pwm_r * MAX_RPM, 1),
        }
        self._pub_lpwm.publish(Float32(data=float(pwm_l)))
        self._pub_rpwm.publish(Float32(data=float(pwm_r)))

    def _cb_lora(self, msg: String):
        try:
            d = json.loads(msg.data)
            self._lora.update(d)
        except Exception:
            pass

    def _cb_auto(self, msg: String):
        try:
            d = json.loads(msg.data)
            self._auto.update(d)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Watchdog: stop motors if cmd_vel is stale
    # ------------------------------------------------------------------
    def _watchdog(self):
        if time.monotonic() - self._last_cmd_vel_t > 1.0:
            self._motors = {"left_rpm": 0.0, "right_rpm": 0.0}
            self._pub_lpwm.publish(Float32(data=0.0))
            self._pub_rpwm.publish(Float32(data=0.0))

    # ------------------------------------------------------------------
    # Telemetry publish
    # ------------------------------------------------------------------
    def _publish_telemetry(self):
        payload = {
            "gps":     self._gps,
            "imu":     self._imu,
            "battery": self._battery,
            "tidal":   self._tidal,
            "motors":  self._motors,
            "arm":     self._arm,
            "lora":    self._lora,
            "auto":    self._auto,
            "ts":      time.time(),
        }
        self._pub_tel.publish(String(data=json.dumps(payload)))


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
