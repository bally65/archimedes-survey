"""
detect_burrow.py
================
YOLOv8-nano based burrow-opening detection for the Archimedes Survey robot.
Targets: 穴口 (Lysiosquillina burrow openings) on sand beach surfaces.

Usage:
    python detect_burrow.py --source 0                    # webcam
    python detect_burrow.py --source image.jpg            # single image
    python detect_burrow.py --source /path/to/video.mp4  # video file
    python detect_burrow.py --source /path/to/images/    # directory

Camera geometry (pinhole model):
    Given camera height h above ground, focal lengths fx/fy, and principal
    point (cx, cy), a detected burrow centre at pixel (u, v) maps to a
    ground-plane position (X, Y) in the robot/camera frame:
        X_fwd  = h * (yn*cos(t) + sin(t)) / (cos(t) - yn*sin(t))
        Y_lat  = h *  xn                  / (cos(t) - yn*sin(t))
    where xn=(u-cx)/fx, yn=(v-cy)/fy, t=tilt_deg in radians.
    (t=0 -> camera points straight down.)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run:  pip install ultralytics")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Pinhole camera intrinsics and mounting geometry.

    Default values correspond to RPi Camera Module 3 (IMX708)
    downscaled to 640x640 inference resolution.
    """
    inference_w: int   = 640
    inference_h: int   = 640
    fx: float          = 800.0 * (640 / 1920)   # ~266.7 px
    fy: float          = 800.0 * (640 / 1080)   # ~474.1 px
    cx: float          = 320.0
    cy: float          = 320.0
    camera_height_m: float = 0.30
    tilt_deg: float        = 0.0

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert inference-space pixel (u, v) to ground-plane (x_fwd, y_lat) metres.
        x_fwd > 0: forward;  y_lat > 0: left.
        """
        h = self.camera_height_m
        tilt_rad = math.radians(self.tilt_deg)
        xn = (u - self.cx) / self.fx
        yn = (v - self.cy) / self.fy
        ct, st = math.cos(tilt_rad), math.sin(tilt_rad)
        denom = ct - yn * st
        if abs(denom) < 1e-6:
            denom = 1e-6
        x_fwd = h * (yn * ct + st) / denom
        y_lat = h * xn / denom
        return (x_fwd, y_lat)


@dataclass
class BurrowDetection:
    """Single detected burrow opening."""
    bbox: Tuple[int, int, int, int]       # (x1,y1,x2,y2) in inference px
    confidence: float
    center_px: Tuple[float, float]        # (u, v) in inference px
    center_world_m: Tuple[float, float]   # (x_fwd, y_lat) metres
    class_id: int = 0
    label: str    = "burrow"

    def to_dict(self) -> dict:
        return {
            "bbox":           list(self.bbox),
            "confidence":     round(self.confidence, 4),
            "center_px":      [round(c, 1) for c in self.center_px],
            "center_world_m": [round(c, 4) for c in self.center_world_m],
            "label":          self.label,
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class BurrowDetector:
    """YOLOv8-nano burrow detector with pinhole ground-plane projection."""

    CLASS_NAMES = {0: "burrow"}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.50,
        calib: Optional[CameraCalibration] = None,
        output_dir: Optional[str] = None,
    ):
        self.conf_threshold = conf_threshold
        self.calib = calib or CameraCalibration()
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[BurrowDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("[BurrowDetector] Model ready.")

    def detect(self, image: np.ndarray) -> List[BurrowDetection]:
        """Run inference on a BGR image. Returns detections sorted by confidence."""
        orig_h, orig_w = image.shape[:2]
        inf_w, inf_h = self.calib.inference_w, self.calib.inference_h

        results = self.model.predict(
            source=image,
            imgsz=(inf_h, inf_w),
            conf=self.conf_threshold,
            verbose=False,
        )

        detections: List[BurrowDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Rescale original px -> inference px
                x1 = x1 * inf_w / orig_w
                y1 = y1 * inf_h / orig_h
                x2 = x2 * inf_w / orig_w
                y2 = y2 * inf_h / orig_h
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                world_xy = self.calib.pixel_to_ground(cx_px, cy_px)
                detections.append(BurrowDetection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=conf,
                    center_px=(cx_px, cy_px),
                    center_world_m=world_xy,
                    class_id=cls_id,
                    label=self.CLASS_NAMES.get(cls_id, f"cls{cls_id}"),
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def annotate(self, image: np.ndarray, detections: List[BurrowDetection]) -> np.ndarray:
        """Draw bounding boxes, confidence, and world-coords on image."""
        vis = image.copy()
        orig_h, orig_w = vis.shape[:2]
        sx = orig_w / self.calib.inference_w
        sy = orig_h / self.calib.inference_h

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            ix1, iy1 = int(x1 * sx), int(y1 * sy)
            ix2, iy2 = int(x2 * sx), int(y2 * sy)
            icx = int(det.center_px[0] * sx)
            icy = int(det.center_px[1] * sy)

            color = (0, 200, 255)
            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.drawMarker(vis, (icx, icy), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)

            xw, yw = det.center_world_m
            label = f"{det.label} {det.confidence:.2f} | ({xw:+.3f}m, {yw:+.3f}m)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.55
            (tw, th), bl = cv2.getTextSize(label, font, fs, 1)
            cv2.rectangle(vis, (ix1, iy1 - th - bl - 4), (ix1 + tw + 4, iy1), color, cv2.FILLED)
            cv2.putText(vis, label, (ix1 + 2, iy1 - bl - 2), font, fs, (0, 0, 0), 1, cv2.LINE_AA)

        hud = f"Burrows: {len(detections)}  h={self.calib.camera_height_m:.2f}m  conf>={self.conf_threshold:.2f}"
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return vis

    def save_annotated(self, annotated: np.ndarray, stem: str, timestamp: Optional[float] = None) -> Path:
        if self.output_dir is None:
            raise RuntimeError("output_dir not configured.")
        ts  = timestamp or time.time()
        out = self.output_dir / f"{stem}_{int(ts * 1000)}.jpg"
        cv2.imwrite(str(out), annotated)
        return out

    def process_image(self, path: str) -> List[BurrowDetection]:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        dets      = self.detect(img)
        annotated = self.annotate(img, dets)
        if self.output_dir:
            saved = self.save_annotated(annotated, Path(path).stem)
            print(f"[BurrowDetector] Saved -> {saved}")
        return dets

    def process_stream(self, source, display: bool = True, max_fps: float = 10.0) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        frame_delay = 1.0 / max_fps
        idx = 0
        print(f"[BurrowDetector] Streaming source={source!r}. Press Q to quit.")
        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                dets      = self.detect(frame)
                annotated = self.annotate(frame, dets)
                if dets:
                    print(f"[{idx:05d}] {len(dets)} burrow(s): " +
                          ", ".join(f"conf={d.confidence:.2f} world=({d.center_world_m[0]:+.3f},{d.center_world_m[1]:+.3f})m" for d in dets))
                if self.output_dir and dets:
                    self.save_annotated(annotated, f"frame_{idx:05d}", t0)
                if display:
                    cv2.imshow("Archimedes Burrow Detection", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                elapsed = time.time() - t0
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                idx += 1
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[BurrowDetector] Done. {idx} frames processed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Archimedes Survey - YOLOv8-nano burrow detector")
    p.add_argument("--source",        default="0")
    p.add_argument("--model",         default="yolov8n.pt")
    p.add_argument("--conf",          type=float, default=0.50)
    p.add_argument("--camera-height", type=float, default=0.30)
    p.add_argument("--tilt-deg",      type=float, default=0.0)
    p.add_argument("--output-dir",    default="output")
    p.add_argument("--no-display",    action="store_true")
    p.add_argument("--fps",           type=float, default=10.0)
    args = p.parse_args()

    calib = CameraCalibration(camera_height_m=args.camera_height, tilt_deg=args.tilt_deg)
    detector = BurrowDetector(model_path=args.model, conf_threshold=args.conf,
                               calib=calib, output_dir=args.output_dir)

    src = Path(args.source)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if src.is_dir():
        imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in image_exts)
        if not imgs:
            sys.exit(f"No images found in {src}")
        for img_path in imgs:
            dets = detector.process_image(str(img_path))
            print(f"  {img_path.name}: {len(dets)} detection(s)",
                  json.dumps([d.to_dict() for d in dets], ensure_ascii=False))
    elif src.is_file() and src.suffix.lower() in image_exts:
        dets = detector.process_image(str(src))
        print(f"\n{len(dets)} detection(s):")
        for d in dets:
            print(" ", json.dumps(d.to_dict(), ensure_ascii=False))
    else:
        try:
            stream_src = int(args.source)
        except ValueError:
            stream_src = args.source
        detector.process_stream(source=stream_src, display=not args.no_display, max_fps=args.fps)


if __name__ == "__main__":
    main()
