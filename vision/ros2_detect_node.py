"""
ros2_detect_node.py
===================
ROS2 node wrapper for the Archimedes Survey burrow detector.

Node     : burrow_detector
Subscribe: /camera/image_raw          (sensor_msgs/Image)
Publish  : /burrow_detections         (std_msgs/String -- JSON)
           /camera/annotated          (sensor_msgs/Image)

Parameters:
    model_path           str   Path to .pt weights       [yolov8n.pt]
    confidence_threshold float Min detection confidence  [0.5]
    camera_height_m      float Camera height above ground [0.30]
    tilt_deg             float Camera tilt from vertical  [0.0]
    process_hz           float Max processing rate (Hz)   [10.0]
    output_dir           str   Save frames dir ('' = off) ['']

Usage:
    ros2 run archimedes_survey ros2_detect_node
    ros2 run archimedes_survey ros2_detect_node \\
        --ros-args -p model_path:=/home/pi/burrow_best.pt \\
                   -p camera_height_m:=0.25 \\
                   -p confidence_threshold:=0.45
"""

import json
import sys
import time
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:
    sys.exit("ROS2 Python packages not found.\n"
             "Source workspace:  source /opt/ros/<distro>/setup.bash")

try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:
    sys.exit("cv_bridge not found.\n"
             "Install:  sudo apt install ros-<distro>-cv-bridge")

try:
    from detect_burrow import BurrowDetector, CameraCalibration
except ImportError:
    sys.exit("detect_burrow.py not on PYTHONPATH.\n"
             "Run from vision/ or add to sys.path.")


class BurrowDetectorNode(Node):
    """ROS2 node: subscribes to /camera/image_raw, publishes detections."""

    def __init__(self) -> None:
        super().__init__("burrow_detector")

        self.declare_parameter("model_path",           "yolov8n.pt")
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("camera_height_m",      0.30)
        self.declare_parameter("tilt_deg",             0.0)
        self.declare_parameter("process_hz",           10.0)
        self.declare_parameter("output_dir",           "")

        model_path = self.get_parameter("model_path").value
        conf       = self.get_parameter("confidence_threshold").value
        height     = self.get_parameter("camera_height_m").value
        tilt       = self.get_parameter("tilt_deg").value
        process_hz = self.get_parameter("process_hz").value
        output_dir = self.get_parameter("output_dir").value or None

        self.get_logger().info(
            f"BurrowDetectorNode: model={model_path!r} conf={conf} "
            f"height={height}m tilt={tilt}deg hz={process_hz}"
        )

        calib = CameraCalibration(camera_height_m=height, tilt_deg=tilt)
        self._detector = BurrowDetector(
            model_path=model_path,
            conf_threshold=conf,
            calib=calib,
            output_dir=output_dir,
        )
        self._bridge = CvBridge()
        self._min_period_s: float = 1.0 / max(process_hz, 0.1)
        self._last_t: float = 0.0

        self._pub_det = self.create_publisher(String, "/burrow_detections", 10)
        self._pub_ann = self.create_publisher(Image,  "/camera/annotated",  10)
        self._sub     = self.create_subscription(
            Image, "/camera/image_raw", self._cb, 10
        )
        self.get_logger().info("Listening on /camera/image_raw ...")

    def _cb(self, msg: Image) -> None:
        now = time.monotonic()
        if now - self._last_t < self._min_period_s:
            return
        self._last_t = now

        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge: {e}")
            return

        try:
            dets = self._detector.detect(cv_img)
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
            return

        # Publish JSON
        payload = {
            "stamp_sec":     msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "frame_id":      msg.header.frame_id,
            "count":         len(dets),
            "detections":    [d.to_dict() for d in dets],
        }
        self._pub_det.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        if dets:
            self.get_logger().info(
                f"Detected {len(dets)} burrow(s): " +
                ", ".join(f"conf={d.confidence:.2f} "
                          f"world=({d.center_world_m[0]:+.3f},{d.center_world_m[1]:+.3f})m"
                          for d in dets)
            )

        # Publish annotated image
        try:
            ann = self._detector.annotate(cv_img, dets)
            ann_msg = self._bridge.cv2_to_imgmsg(ann, encoding="bgr8")
            ann_msg.header = msg.header
            self._pub_ann.publish(ann_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Annotated publish: {e}")

        # Save if configured
        if self._detector.output_dir and dets:
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            try:
                self._detector.save_annotated(ann, "ros_frame", ts)
            except Exception as e:
                self.get_logger().warning(f"Save failed: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BurrowDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
