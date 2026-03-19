#!/usr/bin/env python3
"""
record_cscan.py
===============
訂閱 /acoustic/cscan_volume，將下一筆體積資料儲存為 .npy + .json。

Usage:
    python3 record_cscan.py --label B --count 1 --notes "清晰單洞深35cm" \
                             --output data/cscan_dataset/B/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _ROS2 = True
except ImportError:
    _ROS2 = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label",  required=True,
                   choices=["C", "B", "O", "M"],
                   help="C=Control, B=Burrow, O=Obscured, M=Multiple")
    p.add_argument("--count",  type=int, default=1,
                   help="Number of burrows (ignored for C/O)")
    p.add_argument("--notes",  default="",
                   help="Free-text field notes")
    p.add_argument("--output", required=True,
                   help="Output directory (e.g. data/cscan_dataset/B/)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="Seconds to wait for a volume message (default 30)")
    return p.parse_args()


class VolumeRecorder(Node):
    def __init__(self, args):
        super().__init__("cscan_recorder")
        self._args    = args
        self._received = False
        self._sub = self.create_subscription(
            String, "/acoustic/cscan_volume",
            self._cb, 1)
        self.get_logger().info("Waiting for /acoustic/cscan_volume ...")

    def _cb(self, msg: String):
        if self._received:
            return
        self._received = True
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON parse error: {e}")
            return

        volume = np.array(data["volume"], dtype=np.float32)
        meta   = data.get("meta", {})

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self._args.output, exist_ok=True)
        stem = os.path.join(self._args.output, f"scan_{ts}")

        np.save(f"{stem}.npy", volume)

        record = {
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "location_lat": meta.get("location_lat", None),
            "location_lon": meta.get("location_lon", None),
            "tide_height_m": meta.get("tide_height_m", None),
            "water_temp_c": meta.get("temperature_c", None),
            "label":        self._args.label,
            "burrow_count": self._args.count if self._args.label in ("B", "M") else 0,
            "peak_depth_m": meta.get("peak_depth_m", None),
            "snr_db":       meta.get("snr_db", None),
            "notes":        self._args.notes,
            "volume_shape": list(volume.shape),
        }
        with open(f"{stem}.json", "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        self.get_logger().info(
            f"Saved: {stem}.npy  shape={volume.shape}  label={self._args.label}"
        )


def main():
    args = parse_args()

    if not _ROS2:
        print("ERROR: rclpy not available. Run inside ROS2 environment.", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = VolumeRecorder(args)

    deadline = time.time() + args.timeout
    while not node._received and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node._received:
        node.get_logger().error(
            f"Timeout ({args.timeout}s): no volume received on /acoustic/cscan_volume"
        )
        sys.exit(1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
