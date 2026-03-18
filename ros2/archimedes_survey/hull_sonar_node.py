"""
hull_sonar_node.py
==================
高潮位掃描節點：船底向下魚探換能器，漲潮時掃描底部回波異常（洞穴熱區）。

工作模式：
  - 漲潮期（水深 > 0.1m）：船浮在水面，向下掃描底部
  - 底部有洞穴開口 → 回波強度異常（強反射點 or 不規則散射）
  - 記錄 GPS + 回波強度 → 產生熱區 GeoJSON（供退潮後地面任務使用）

硬體選項（任選一種，成本遞增）：
  A. 釣魚魚探（UART NMEA-0183）：NT$800~1,500，釣具行直接買
     → 深度輸出為 $SDDBT 或 $SDDBS 句型
  B. 現有 ultrasound 電路（200kHz 船底安裝）：NT$2,500
     → 與 ultrasound_node.py 共用驅動邏輯
  C. 模擬模式：開發測試用

Topics Published:
  /hull_sonar/depth_m       (Float32)   — 目前底部深度 (m)
  /hull_sonar/echo_strength (Float32)   — 回波強度 0~1（大 = 異常/洞穴可能）
  /hull_sonar/hotspot       (String)    — JSON：偵測到洞穴熱區
  /hull_sonar/scanline      (String)    — JSON：掃描線資料（GPS+深度+強度）

Topics Subscribed:
  /gps/fix                  (NavSatFix) — GPS 位置（與聲納同步）
  /imu/data                 (Imu)       — 船體傾角修正深度量測

Run:
  ros2 run archimedes_survey hull_sonar_node
  ros2 run archimedes_survey hull_sonar_node --ros-args -p simulate:=true
"""

import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String
    from sensor_msgs.msg import NavSatFix, Imu
except ImportError:
    sys.exit("ROS2 not found.")

try:
    import serial   # pyserial，for NMEA fishfinder
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

# ---------------------------------------------------------------------------
# 參數
# ---------------------------------------------------------------------------
HULL_DRAFT_M   = 0.08      # 船底到水面距離 (m)，轉換水深 → 底部深度
MIN_WATER_M    = 0.10      # 最小可量測水深 (m)
SCAN_RATE_HZ   = 2.0       # 掃描頻率 (Hz)

# 洞穴熱區偵測閾值
ECHO_ANOMALY_THRESHOLD = 0.65   # 回波強度超過此值判定為異常
HOTSPOT_MIN_HITS       = 3      # 連續 N 點異常才發布熱區
HOTSPOT_RADIUS_M       = 0.5    # 熱區合併半徑 (m，同一熱區內的點合併)

# NMEA 串口設定（A 路線魚探）
NMEA_PORT      = "/dev/ttyUSB1"
NMEA_BAUD      = 4800


# ---------------------------------------------------------------------------
# NMEA 解析
# ---------------------------------------------------------------------------
def parse_nmea_depth(line: str) -> tuple:
    """
    解析 NMEA 深度句：
      $SDDBT,xx.x,f,xx.x,M,xx.x,F*hh  (depth below transducer)
      $SDDBS,xx.x,f,xx.x,M,xx.x,F*hh  (depth below surface)
    回傳 (depth_m, valid)
    """
    line = line.strip()
    if not (line.startswith("$SDDBT") or line.startswith("$SDDBS")):
        return 0.0, False
    try:
        parts = line.split(",")
        # field index 3 = metres value
        depth_m = float(parts[3])
        return depth_m, True
    except (IndexError, ValueError):
        return 0.0, False


# ---------------------------------------------------------------------------
# 模擬器（開發測試）
# ---------------------------------------------------------------------------
class SimulatedSonar:
    """模擬漲潮期在洞穴分布區掃描的回波資料。"""

    def __init__(self):
        self._t     = 0.0
        # 模擬 3 個洞穴熱區（相對 GPS 座標，m）
        self._zones = [(10, 5), (-8, 3), (2, -12)]

    def read(self, boat_x: float = 0.0, boat_y: float = 0.0) -> tuple:
        """回傳 (depth_m, echo_strength)。"""
        self._t += 1.0 / SCAN_RATE_HZ
        # 基底水深 (模擬退潮梯度)
        base_depth = 0.4 + 0.2 * math.sin(self._t * 0.05)

        # 若附近有模擬洞穴熱區 → 增強回波強度
        min_dist = min(
            math.sqrt((boat_x - zx)**2 + (boat_y - zy)**2)
            for zx, zy in self._zones
        )
        if min_dist < 1.5:
            # 洞穴區：回波較強 + 深度略有起伏（底部不平整）
            strength = 0.8 - min_dist * 0.1 + np.random.normal(0, 0.05)
            depth    = base_depth + np.random.normal(0, 0.03)
        else:
            # 平坦泥灘：回波均勻，強度低
            strength = 0.25 + np.random.normal(0, 0.04)
            depth    = base_depth + np.random.normal(0, 0.01)

        return max(MIN_WATER_M, depth), max(0.0, min(1.0, strength))


# ---------------------------------------------------------------------------
# 主節點
# ---------------------------------------------------------------------------
class HullSonarNode(Node):

    def __init__(self):
        super().__init__("hull_sonar_node")

        # 參數
        self.declare_parameter("simulate",   True)
        self.declare_parameter("nmea_port",  NMEA_PORT)
        self.declare_parameter("nmea_baud",  NMEA_BAUD)
        self.declare_parameter("scan_hz",    SCAN_RATE_HZ)
        self.declare_parameter("anomaly_threshold", ECHO_ANOMALY_THRESHOLD)

        simulate  = self.get_parameter("simulate").value
        port      = self.get_parameter("nmea_port").value
        baud      = self.get_parameter("nmea_baud").value
        scan_hz   = float(self.get_parameter("scan_hz").value)
        self._thresh = float(self.get_parameter("anomaly_threshold").value)

        # 硬體初始化
        self._serial = None
        if not simulate and _SERIAL_AVAILABLE:
            try:
                self._serial = serial.Serial(port, baud, timeout=0.5)
                self.get_logger().info(f"NMEA fishfinder: {port} @ {baud}")
            except serial.SerialException as e:
                self.get_logger().warn(f"Serial open failed ({e}), using simulate")
                simulate = True
        elif not simulate:
            self.get_logger().warn("pyserial not installed, using simulate")
            simulate = True

        self._sim = SimulatedSonar() if simulate else None
        self.get_logger().info(
            f"HullSonar init: {'SIMULATE' if simulate else 'NMEA'}, "
            f"anomaly_thresh={self._thresh:.2f}"
        )

        # 狀態
        self._lat: float = None
        self._lon: float = None
        self._pitch_rad: float = 0.0
        self._roll_rad:  float = 0.0
        self._anomaly_streak = 0          # 連續異常次數
        self._hotspots: list = []         # 已記錄熱區
        self._scanline: list = []         # 本次掃描線
        self._mission_start = datetime.now().isoformat()
        self._sim_x = 0.0                 # 模擬位置（用於 simulate 模式）
        self._sim_y = 0.0

        # 熱區輸出路徑
        self._hotspot_path = (
            Path.home() / "archimedes_missions" /
            f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_hotspots.geojson"
        )
        self._hotspot_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_hotspots()

        # Subscribers
        self.create_subscription(NavSatFix, "/gps/fix", self._cb_gps,  10)
        self.create_subscription(Imu, "/imu/data",      self._cb_imu,  10)

        # Publishers
        self._pub_depth    = self.create_publisher(Float32, "/hull_sonar/depth_m",       10)
        self._pub_strength = self.create_publisher(Float32, "/hull_sonar/echo_strength", 10)
        self._pub_hotspot  = self.create_publisher(String,  "/hull_sonar/hotspot",        5)
        self._pub_scanline = self.create_publisher(String,  "/hull_sonar/scanline",       5)

        # 掃描定時器
        self.create_timer(1.0 / scan_hz, self._scan_tick)

    # -----------------------------------------------------------------------
    def _cb_gps(self, msg: NavSatFix):
        if msg.status.status >= 0:
            self._lat = msg.latitude
            self._lon = msg.longitude

    def _cb_imu(self, msg: Imu):
        """取船體 pitch/roll 修正深度量測。"""
        # 簡化：取 angular_velocity 作為傾角估計（實際應用四元數）
        self._pitch_rad = msg.angular_velocity.y * 0.01
        self._roll_rad  = msg.angular_velocity.x * 0.01

    # -----------------------------------------------------------------------
    def _scan_tick(self):
        """定時觸發一次聲納量測。"""
        # 更新模擬位置（直線掃描，每次前進 0.1m）
        if self._sim is not None:
            self._sim_x += 0.1
            depth_m, strength = self._sim.read(self._sim_x, self._sim_y)
        else:
            depth_m, strength = self._read_nmea()
            if depth_m <= 0:
                return

        # 船體傾角修正（cos補償）
        tilt = math.sqrt(self._pitch_rad**2 + self._roll_rad**2)
        depth_m_corr = depth_m * math.cos(tilt) if tilt < 0.5 else depth_m

        # 發布深度與強度
        self._pub_depth.publish(Float32(data=float(depth_m_corr)))
        self._pub_strength.publish(Float32(data=float(strength)))

        # 記錄掃描線點
        pt = {
            "lat":      self._lat,
            "lon":      self._lon,
            "depth_m":  round(depth_m_corr, 3),
            "strength": round(strength, 3),
            "ts":       time.time(),
        }
        self._scanline.append(pt)
        self._pub_scanline.publish(String(data=json.dumps(pt)))

        # 異常偵測（洞穴熱區判定）
        if strength >= self._thresh:
            self._anomaly_streak += 1
            self.get_logger().debug(
                f"Echo anomaly #{self._anomaly_streak}: "
                f"depth={depth_m_corr:.2f}m strength={strength:.2f}"
            )
            if self._anomaly_streak >= HOTSPOT_MIN_HITS:
                self._register_hotspot(depth_m_corr, strength)
                self._anomaly_streak = 0
        else:
            self._anomaly_streak = max(0, self._anomaly_streak - 1)

    # -----------------------------------------------------------------------
    def _read_nmea(self) -> tuple:
        """從 NMEA 串口讀取深度。"""
        if self._serial is None:
            return 0.0, 0.0
        try:
            line = self._serial.readline().decode("ascii", errors="ignore")
            depth_m, valid = parse_nmea_depth(line)
            if valid:
                # 估計回波強度：無原始 ADC，用深度穩定性代理
                strength = 0.5  # placeholder
                return depth_m, strength
        except Exception:
            pass
        return 0.0, 0.0

    # -----------------------------------------------------------------------
    def _register_hotspot(self, depth_m: float, strength: float):
        """記錄洞穴熱區（合併相近的點）。"""
        if self._lat is None:
            return

        # 檢查是否與現有熱區太近（合併）
        for hs in self._hotspots:
            dist = _haversine(hs["lat"], hs["lon"], self._lat, self._lon)
            if dist < HOTSPOT_RADIUS_M:
                # 更新既有熱區（取最大強度）
                if strength > hs["max_strength"]:
                    hs["max_strength"] = round(strength, 3)
                    hs["depth_m"] = round(depth_m, 3)
                hs["hit_count"] += 1
                self._write_hotspots()
                return

        # 新熱區
        hs = {
            "id":           len(self._hotspots),
            "lat":          self._lat,
            "lon":          self._lon,
            "depth_m":      round(depth_m, 3),
            "max_strength": round(strength, 3),
            "hit_count":    1,
            "timestamp":    datetime.now().isoformat(),
        }
        self._hotspots.append(hs)
        self._write_hotspots()

        msg = json.dumps(hs)
        self._pub_hotspot.publish(String(data=msg))
        self.get_logger().info(
            f"Hotspot #{hs['id']} detected: "
            f"lat={self._lat:.6f} lon={self._lon:.6f} "
            f"depth={depth_m:.2f}m strength={strength:.2f}"
        )

    # -----------------------------------------------------------------------
    def _write_hotspots(self):
        features = []
        for hs in self._hotspots:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [hs["lon"], hs["lat"]],
                },
                "properties": {
                    "id":           hs["id"],
                    "depth_m":      hs["depth_m"],
                    "max_strength": hs["max_strength"],
                    "hit_count":    hs["hit_count"],
                    "timestamp":    hs["timestamp"],
                    "survey_phase": "high_tide_scan",
                    "note":         "需退潮後地面確認",
                },
            })
        fc = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "mission_start": self._mission_start,
                "total_hotspots": len(features),
                "method": "hull-mounted downward sonar (200kHz)",
            },
        }
        self._hotspot_path.write_text(json.dumps(fc, indent=2))


# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = HullSonarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Hull sonar shutdown. Hotspots saved to {node._hotspot_path}"
        )
    finally:
        if node._serial:
            node._serial.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
