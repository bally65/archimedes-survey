"""
mission_logger.py
=================
ROS2 node: records GPS tracks and burrow detections to GeoJSON files.

Output files (in ~/archimedes_missions/):
  YYYY-MM-DD_HHMMSS_track.geojson    -- GPS LineString (full path)
  YYYY-MM-DD_HHMMSS_burrows.geojson  -- burrow Points with metadata

GeoJSON is directly importable into:
  - QGIS (drag & drop)
  - Google My Maps (import)
  - geojson.io (paste)
  - Any GIS software

Subscribe:
  /gps/fix            (sensor_msgs/NavSatFix)
  /burrow_detections  (std_msgs/String -- JSON)
  /auto_status        (std_msgs/String -- JSON)

Run:
    ros2 run archimedes_survey mission_logger
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import NavSatFix
    from std_msgs.msg import String
except ImportError:
    sys.exit("ROS2 not found.")

MISSION_DIR = Path.home() / "archimedes_missions"
TRACK_INTERVAL_S = 2.0      # record GPS position every N seconds
MIN_MOVE_M       = 0.5      # skip track point if moved less than this


def _haversine(lat1, lon1, lat2, lon2) -> float:
    import math
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MissionLoggerNode(Node):
    def __init__(self):
        super().__init__("mission_logger")

        MISSION_DIR.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._track_path  = MISSION_DIR / f"{ts}_track.geojson"
        self._burrow_path = MISSION_DIR / f"{ts}_burrows.geojson"

        self._track_coords: list[list[float]] = []   # [[lon, lat], ...]
        self._burrows:      list[dict] = []
        self._last_track_t  = 0.0
        self._last_lat      = None
        self._last_lon      = None
        self._mission_start = datetime.now().isoformat()
        self._burrow_count  = 0
        # Y-burrow pairing: Austinogebia edulis has 2 openings per individual (21-26cm apart)
        # Track openings in metric coords to pair them before counting individuals
        self._opening_positions: list[tuple[float, float]] = []  # (world_x_m, world_y_m)
        self._Y_PAIR_DIST_M = 0.30   # openings within 30cm → same individual
        # 最新聲學量測結果（由 acoustic_processor 發布）
        self._last_acoustic: dict = {}
        # 最新內視鏡確認結果（由 endoscope_node 發布）
        self._last_endoscope: dict = {}

        # Init empty GeoJSON files
        self._write_track()
        self._write_burrows()

        self.create_subscription(NavSatFix, "/gps/fix",              self._cb_gps,      10)
        self.create_subscription(String,    "/burrow_detections",   self._cb_burrow,   10)
        self.create_subscription(String,    "/auto_status",         self._cb_status,   10)
        self.create_subscription(String,    "/ultrasound/burrow_3d", self._cb_acoustic, 10)
        # 內視鏡洞穴確認（/endoscope/result: JSON {"occupied": bool, "confidence": 0~1}）
        self.create_subscription(String,    "/endoscope/result",    self._cb_endoscope, 10)

        self.get_logger().info(
            f"MissionLogger started\n"
            f"  Track:  {self._track_path}\n"
            f"  Burrows:{self._burrow_path}"
        )

    # ------------------------------------------------------------------
    def _cb_gps(self, msg: NavSatFix):
        if msg.status.status < 0:   # no fix
            return
        lat, lon = msg.latitude, msg.longitude
        now = time.monotonic()

        if now - self._last_track_t < TRACK_INTERVAL_S:
            return

        if self._last_lat is not None:
            dist = _haversine(self._last_lat, self._last_lon, lat, lon)
            if dist < MIN_MOVE_M:
                return

        self._track_coords.append([lon, lat])
        self._last_lat    = lat
        self._last_lon    = lon
        self._last_track_t = now
        self._write_track()

    def _is_new_individual(self, wx: float, wy: float) -> bool:
        """Return True if (wx,wy) is >Y_PAIR_DIST_M from all known openings.
        Austinogebia edulis has Y-shaped burrows with 2 openings 21-26cm apart.
        Openings within 30cm are treated as the same individual (not double-counted).
        """
        import math
        for ox, oy in self._opening_positions:
            if math.hypot(wx - ox, wy - oy) < self._Y_PAIR_DIST_M:
                return False  # paired with existing opening → same individual
        return True

    def _cb_burrow(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception:
            return

        if not d.get("detections") or self._last_lat is None:
            return

        for det in d["detections"]:
            wx = round(det.get("center_world_m", [0, 0])[0], 3)
            wy = round(det.get("center_world_m", [0, 0])[1], 3)
            is_new = self._is_new_individual(wx, wy)
            self._opening_positions.append((wx, wy))
            entry = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [self._last_lon, self._last_lat],
                },
                "properties": {
                    "id":         self._burrow_count,
                    "timestamp":  datetime.now().isoformat(),
                    "confidence": round(det.get("confidence", 0), 3),
                    "world_x_m":  wx,
                    "world_y_m":  wy,
                    "frame_id":   d.get("frame_id", ""),
                    # Y-burrow pairing: is this opening a new individual or paired with prior?
                    "is_new_individual": is_new,
                    # 聲學量測（若有，future use）
                    "burrow_depth_m":         self._last_acoustic.get("burrow_depth_m"),
                    "burrow_angle_deg":       self._last_acoustic.get("burrow_angle_deg"),
                    "ultrasound_confidence":  self._last_acoustic.get("ultrasound_confidence"),
                    # 內視鏡確認（優先用於判定活動洞穴）
                    "occupied":               self._last_endoscope.get("occupied"),
                    "endoscope_confidence":   self._last_endoscope.get("confidence"),
                },
            }
            self._burrows.append(entry)
            self._burrow_count += 1
            self._write_burrows()
            pair_note = "NEW individual" if is_new else "paired opening (same individual)"
            self.get_logger().info(
                f"Opening #{self._burrow_count} logged [{pair_note}]: "
                f"lat={self._last_lat:.6f} lon={self._last_lon:.6f} "
                f"conf={det.get('confidence', 0):.2f}"
            )

    def _cb_acoustic(self, msg: String):
        """接收 acoustic_processor 的洞穴深度/傾角，暫存供下次 burrow 記錄用。"""
        try:
            d = json.loads(msg.data)
            self._last_acoustic = d.get("geojson_extra", d)
        except Exception:
            pass

    def _cb_endoscope(self, msg: String):
        """接收內視鏡確認結果（occupied: bool, confidence: 0~1）。"""
        try:
            self._last_endoscope = json.loads(msg.data)
        except Exception:
            pass

    def _cb_status(self, msg: String):
        try:
            d = json.loads(msg.data)
            if d.get("mode") in ("IDLE",) and d.get("burrows", 0) > 0:
                self.get_logger().info(
                    f"Mission complete: {d['burrows']} burrows found, "
                    f"saved to {self._burrow_path}"
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _write_track(self):
        coords = self._track_coords
        if len(coords) < 2:
            geom = {"type": "LineString", "coordinates": coords or [[0, 0], [0, 0]]}
        else:
            geom = {"type": "LineString", "coordinates": coords}

        fc = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "mission_start": self._mission_start,
                    "points":        len(coords),
                },
            }],
        }
        self._track_path.write_text(json.dumps(fc, indent=2))

    def _write_burrows(self):
        # Austinogebia edulis has Y-shaped burrows with 2 openings per individual
        total_openings = len(self._burrows)
        total_individuals = sum(
            1 for b in self._burrows
            if b["properties"].get("is_new_individual", True)
        )
        fc = {
            "type": "FeatureCollection",
            "features": self._burrows,
            "metadata": {
                "mission_start":     self._mission_start,
                "total_openings":    total_openings,
                "total_individuals": total_individuals,  # openings paired, ÷2 logic applied
                "species_target":    "Austinogebia edulis",
                "note": "Y-burrow: 2 openings per individual (21-26cm apart). "
                        "is_new_individual=false means paired with prior opening.",
            },
        }
        self._burrow_path.write_text(json.dumps(fc, indent=2))


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = MissionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down — GeoJSON saved.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
