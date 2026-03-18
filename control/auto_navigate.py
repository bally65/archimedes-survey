"""
auto_navigate.py
================
Autonomous navigation ROS2 node for Archimedes Survey robot.

Modes:
  IDLE      -- waiting for target
  GOTO      -- GPS waypoint navigation (proportional controller)
  SURVEY    -- lawnmower pattern around target (burrow search)
  RTH       -- return to home (launch position)
  EMERGENCY -- tidal alert or low battery: forced RTH at max speed

Subscribe:
  /gps/fix            (sensor_msgs/NavSatFix)
  /imu/data           (sensor_msgs/Imu)
  /battery/voltage    (std_msgs/Float32)
  /tidal/range_cm     (std_msgs/Float32)
  /burrow_detections  (std_msgs/String)
  /auto_command       (std_msgs/String)  -- from web dashboard

Publish:
  /cmd_vel            (geometry_msgs/Twist)
  /arm_command        (std_msgs/String)
  /auto_status        (std_msgs/String -- JSON)

Run:
    ros2 run archimedes_survey auto_navigate
"""

import json
import math
import sys
import time
from enum import Enum, auto

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import NavSatFix, Imu
    from std_msgs.msg import Float32, String
except ImportError:
    sys.exit("ROS2 not found.")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ARRIVE_RADIUS_M  = 1.0    # within this → waypoint reached
SURVEY_STEP_M    = 0.8    # lawnmower row spacing
SURVEY_ROWS      = 5      # number of lawnmower rows
MAX_LINEAR_MPS   = 0.088  # max straight-line speed (m/s) from simulation
MAX_ANGULAR_RPS  = 0.35   # max rotation speed (rad/s)
KP_HEADING       = 1.2    # proportional heading gain
TIDAL_ABORT_CM   = 10.0   # tidal water level to abort mission
LOW_BATT_PCT     = 20.0   # battery % to trigger RTH
CTRL_HZ          = 5.0    # control loop rate


class Mode(Enum):
    IDLE      = auto()
    GOTO      = auto()
    SURVEY    = auto()
    RTH       = auto()
    EMERGENCY = auto()


# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    """Distance in metres between two GPS coords."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1, lon1, lat2, lon2):
    """Bearing in degrees (0=N, 90=E)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _offset_gps(lat, lon, bearing_deg, dist_m):
    """New GPS position given start, bearing, distance."""
    R = 6_371_000.0
    d = dist_m / R
    b = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b)*math.sin(d)*math.cos(lat1),
                              math.cos(d) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


# ---------------------------------------------------------------------------
class AutoNavigateNode(Node):
    def __init__(self):
        super().__init__("auto_navigate")

        self._mode    = Mode.IDLE
        self._lat     = 0.0
        self._lon     = 0.0
        self._yaw     = 0.0    # degrees, 0=N
        self._gps_fix = False
        self._battery_pct = 100.0
        self._tidal_cm    = 0.0

        self._home_lat  = None
        self._home_lon  = None

        # GOTO / SURVEY waypoints
        self._waypoints: list[tuple[float, float]] = []
        self._wp_idx    = 0

        # Burrow found list
        self._burrows: list[dict] = []

        # --- publishers ---
        self._pub_vel    = self.create_publisher(Twist,  "/cmd_vel",     10)
        self._pub_arm    = self.create_publisher(String, "/arm_command", 10)
        self._pub_status = self.create_publisher(String, "/auto_status", 10)

        # --- subscribers ---
        self.create_subscription(NavSatFix, "/gps/fix",           self._cb_gps,     10)
        self.create_subscription(Imu,       "/imu/data",          self._cb_imu,     10)
        self.create_subscription(Float32,   "/battery/voltage",   self._cb_battery, 10)
        self.create_subscription(Float32,   "/tidal/range_cm",    self._cb_tidal,   10)
        self.create_subscription(String,    "/burrow_detections", self._cb_burrow,  10)
        self.create_subscription(String,    "/auto_command",      self._cb_cmd,     10)

        # --- control timer ---
        self.create_timer(1.0 / CTRL_HZ, self._control_loop)

        self.get_logger().info("AutoNavigateNode started")

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------
    def _cb_gps(self, msg: NavSatFix):
        self._gps_fix = msg.status.status >= 0
        if self._gps_fix:
            self._lat = msg.latitude
            self._lon = msg.longitude
            if self._home_lat is None:
                self._home_lat = self._lat
                self._home_lon = self._lon
                self.get_logger().info(
                    f"Home set: {self._home_lat:.6f}, {self._home_lon:.6f}")

    def _cb_imu(self, msg: Imu):
        q = msg.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        # Convert ROS yaw (CCW from East) to compass (CW from North)
        self._yaw = (90 - math.degrees(math.atan2(siny, cosy))) % 360

    def _cb_battery(self, msg: Float32):
        v = msg.data
        self._battery_pct = max(0, min(100,
            (v - 10.5) / (12.6 - 10.5) * 100))
        if self._battery_pct < LOW_BATT_PCT and self._mode not in (Mode.RTH, Mode.IDLE):
            self.get_logger().warn("Low battery -- initiating RTH")
            self._start_rth()

    def _cb_tidal(self, msg: Float32):
        self._tidal_cm = msg.data
        if msg.data >= TIDAL_ABORT_CM and self._mode not in (Mode.EMERGENCY, Mode.IDLE):
            self.get_logger().error(f"TIDAL ALERT {msg.data:.1f}cm -- EMERGENCY RTH")
            self._mode = Mode.EMERGENCY

    def _cb_burrow(self, msg: String):
        try:
            d = json.loads(msg.data)
            for det in d.get("detections", []):
                entry = {
                    "lat":  self._lat,
                    "lon":  self._lon,
                    "conf": det.get("confidence", 0),
                    "world": det.get("center_world_m", [0, 0]),
                    "ts":   time.time(),
                }
                self._burrows.append(entry)
                self.get_logger().info(
                    f"Burrow found: lat={self._lat:.6f} lon={self._lon:.6f} "
                    f"conf={entry['conf']:.2f}")
        except Exception:
            pass

    def _cb_cmd(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except Exception:
            return

        if not cmd.get("enabled", False):
            self._stop()
            self._mode = Mode.IDLE
            return

        if "target_lat" in cmd and "target_lon" in cmd:
            self._set_goto(cmd["target_lat"], cmd["target_lon"])
        elif cmd.get("mode") == "survey" and "center_lat" in cmd:
            self._set_survey(cmd["center_lat"], cmd["center_lon"])
        elif cmd.get("mode") == "rth":
            self._start_rth()

    # ------------------------------------------------------------------
    # Mode setters
    # ------------------------------------------------------------------
    def _set_goto(self, lat, lon):
        self._waypoints = [(lat, lon)]
        self._wp_idx    = 0
        self._mode      = Mode.GOTO
        self.get_logger().info(f"GOTO: {lat:.6f}, {lon:.6f}")

    def _set_survey(self, center_lat, center_lon):
        """Generate lawnmower waypoints around a center point."""
        wps = []
        half = SURVEY_ROWS // 2
        for row in range(-half, half + 1):
            lat0, lon0 = _offset_gps(center_lat, center_lon, 90, row * SURVEY_STEP_M)
            lat1, lon1 = _offset_gps(lat0, lon0, 0 if row % 2 == 0 else 180,
                                     SURVEY_STEP_M * (SURVEY_ROWS + 1))
            if row % 2 == 0:
                wps += [(lat0, lon0), (lat1, lon1)]
            else:
                wps += [(lat1, lon1), (lat0, lon0)]
        self._waypoints = wps
        self._wp_idx    = 0
        self._mode      = Mode.SURVEY
        self.get_logger().info(f"SURVEY: {len(wps)} waypoints around "
                               f"{center_lat:.6f}, {center_lon:.6f}")
        # Deploy arm during survey
        self._pub_arm.publish(String(data=json.dumps({"type": "arm", "pose": "deploy"})))

    def _start_rth(self):
        if self._home_lat is None:
            self.get_logger().warn("No home position recorded yet")
            self._stop()
            self._mode = Mode.IDLE
            return
        self._set_goto(self._home_lat, self._home_lon)
        self._mode = Mode.RTH
        self.get_logger().info("RTH initiated")

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def _control_loop(self):
        if self._mode == Mode.IDLE:
            return

        if self._mode == Mode.EMERGENCY:
            self._pub_vel.publish(Twist())   # stop first
            self._start_rth()
            self._mode = Mode.EMERGENCY      # keep EMERGENCY flag
            return

        if self._mode in (Mode.GOTO, Mode.SURVEY, Mode.RTH):
            if not self._gps_fix:
                self.get_logger().warn("No GPS fix, stopping")
                self._stop()
                return
            if self._wp_idx >= len(self._waypoints):
                self._on_mission_complete()
                return
            self._navigate_to(*self._waypoints[self._wp_idx])

        self._publish_status()

    def _navigate_to(self, target_lat, target_lon):
        dist    = _haversine(self._lat, self._lon, target_lat, target_lon)
        bearing = _bearing(self._lat, self._lon, target_lat, target_lon)

        if dist < ARRIVE_RADIUS_M:
            self.get_logger().info(
                f"Waypoint {self._wp_idx+1}/{len(self._waypoints)} reached")
            self._wp_idx += 1
            if self._wp_idx >= len(self._waypoints):
                self._on_mission_complete()
            return

        # Heading error [-180, +180]
        heading_err = (bearing - self._yaw + 540) % 360 - 180

        # Speed: slow down when close or misaligned
        speed_factor = min(1.0, dist / 3.0) * max(0.2, 1.0 - abs(heading_err) / 90.0)
        linear_v  = MAX_LINEAR_MPS * speed_factor
        angular_v = max(-MAX_ANGULAR_RPS,
                        min(MAX_ANGULAR_RPS, math.radians(heading_err) * KP_HEADING))

        twist = Twist()
        twist.linear.x  = linear_v
        twist.angular.z = angular_v
        self._pub_vel.publish(twist)

    def _on_mission_complete(self):
        self._stop()
        if self._mode in (Mode.RTH, Mode.EMERGENCY):
            self.get_logger().info("Home reached.")
            # Stow arm
            self._pub_arm.publish(String(data=json.dumps({"type": "arm", "pose": "stow"})))
        elif self._mode == Mode.SURVEY:
            self.get_logger().info(
                f"Survey complete. {len(self._burrows)} burrows found.")
        self._mode = Mode.IDLE
        self._publish_status()

    def _stop(self):
        self._pub_vel.publish(Twist())

    def _publish_status(self):
        status = {
            "mode":      self._mode.name,
            "wp_idx":    self._wp_idx,
            "wp_total":  len(self._waypoints),
            "burrows":   len(self._burrows),
            "battery":   round(self._battery_pct, 1),
            "tidal_cm":  round(self._tidal_cm, 1),
        }
        self._pub_status.publish(String(data=json.dumps(status)))


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = AutoNavigateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
