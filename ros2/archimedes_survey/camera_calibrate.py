"""
camera_calibrate.py
===================
棋盤格相機標定腳本，輸出 fx/fy/cx/cy + 畸變係數，
並自動更新 detect_burrow.py 中的 CameraCalibration 預設值。

使用方式：
  1. 列印棋盤格（或在螢幕上顯示）：
       python camera_calibrate.py --gen-board
     → 輸出 calibration_board_9x6.png

  2. 擺好相機，對準棋盤格從不同角度拍攝至少 20 張：
       python camera_calibrate.py --capture --count 30
     → 儲存到 ~/calib_images/

  3. 執行標定：
       python camera_calibrate.py --calibrate
     → 輸出 camera_calib.json + 更新 detect_burrow.py 中的預設值

棋盤格規格：9×6 內角點，格子大小 25mm（可列印在 A4 紙）
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("pip install opencv-python numpy")

# ---------------------------------------------------------------------------
BOARD_W    = 9        # 內角點數（水平）
BOARD_H    = 6        # 內角點數（垂直）
SQUARE_MM  = 25.0     # 格子大小 mm
CALIB_DIR  = Path.home() / "calib_images"
CALIB_JSON = Path("camera_calib.json")
DETECT_PY  = Path(__file__).parent.parent.parent / "vision" / "detect_burrow.py"


# ---------------------------------------------------------------------------
def gen_board():
    """生成棋盤格圖片（用 OpenCV 內建）"""
    try:
        board = cv2.aruco.CharucoBoard(
            (BOARD_W + 1, BOARD_H + 1),
            SQUARE_MM / 1000.0,
            SQUARE_MM * 0.8 / 1000.0,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
        )
        img = board.generateImage((900, 600), marginSize=10)
        cv2.imwrite("calibration_board_9x6.png", img)
        print("Saved: calibration_board_9x6.png (print at 100% scale, no margins)")
    except AttributeError:
        # Fallback: classic checkerboard
        img = np.zeros(((BOARD_H + 1) * 80, (BOARD_W + 1) * 80), dtype=np.uint8)
        for r in range(BOARD_H + 1):
            for c in range(BOARD_W + 1):
                if (r + c) % 2 == 0:
                    img[r*80:(r+1)*80, c*80:(c+1)*80] = 255
        cv2.imwrite("calibration_board_9x6.png", img)
        print("Saved: calibration_board_9x6.png")


# ---------------------------------------------------------------------------
def capture(count: int):
    """開啟相機，按空白鍵拍照，拍滿 count 張後退出"""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    captured = 0
    print(f"Press SPACE to capture ({count} needed), Q to quit")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while captured < count:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, (BOARD_W, BOARD_H), corners, found)
            cv2.putText(disp, "Board FOUND - press SPACE",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(disp, "Board not found",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(disp, f"{captured}/{count} captured",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Calibration Capture", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            fname = CALIB_DIR / f"calib_{captured:03d}.jpg"
            cv2.imwrite(str(fname), frame)
            captured += 1
            print(f"  Saved {fname}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured} images to {CALIB_DIR}")


# ---------------------------------------------------------------------------
def calibrate():
    """從 ~/calib_images/ 讀圖，執行標定，輸出 JSON"""
    imgs = sorted(CALIB_DIR.glob("*.jpg")) + sorted(CALIB_DIR.glob("*.png"))
    if len(imgs) < 10:
        sys.exit(f"Need at least 10 images in {CALIB_DIR}, found {len(imgs)}")

    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM

    obj_pts = []
    img_pts = []
    img_shape = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Processing {len(imgs)} images...")
    good = 0
    for path in imgs:
        img  = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]   # (W, H)

        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)
            good += 1
            print(f"  OK: {path.name}")
        else:
            print(f"  SKIP (board not found): {path.name}")

    print(f"\n{good}/{len(imgs)} images usable.")
    if good < 8:
        sys.exit("Need at least 8 valid images. Recapture with better angles.")

    print("Running calibration...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_shape, None, None)
    print(f"RMS reprojection error: {rms:.4f} px  (should be < 1.0)")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    k1, k2, p1, p2, k3 = [float(x) for x in dist.flatten()[:5]]

    result = {
        "image_width":  img_shape[0],
        "image_height": img_shape[1],
        "fx": fx, "fy": fy,
        "cx": cx, "cy": cy,
        "dist": {"k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3},
        "rms_px": round(rms, 4),
        "board": f"{BOARD_W}x{BOARD_H}_{SQUARE_MM}mm",
    }
    CALIB_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nCalibration saved: {CALIB_JSON}")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    # Patch detect_burrow.py with real values
    _patch_detect_burrow(fx, fy, cx, cy, img_shape)
    return result


# ---------------------------------------------------------------------------
def _patch_detect_burrow(fx, fy, cx, cy, img_shape):
    """Update CameraCalibration defaults in detect_burrow.py"""
    if not DETECT_PY.exists():
        print(f"WARNING: {DETECT_PY} not found, skipping patch")
        return

    src = DETECT_PY.read_text(encoding="utf-8")

    # Replace fx, fy lines in CameraCalibration dataclass
    patterns = [
        (r"(fx:\s*float\s*=\s*)[\d.]+", rf"\g<1>{fx:.2f}"),
        (r"(fy:\s*float\s*=\s*)[\d.]+", rf"\g<1>{fy:.2f}"),
        (r"(cx:\s*float\s*=\s*)[\d.]+", rf"\g<1>{cx:.2f}"),
        (r"(cy:\s*float\s*=\s*)[\d.]+", rf"\g<1>{cy:.2f}"),
    ]
    patched = src
    for pat, rep in patterns:
        patched = re.sub(pat, rep, patched)

    if patched != src:
        DETECT_PY.write_text(patched, encoding="utf-8")
        print(f"Patched {DETECT_PY} with calibrated values")
    else:
        print(f"WARNING: Could not patch {DETECT_PY} automatically.")
        print(f"  Please manually set: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")


# ---------------------------------------------------------------------------
def show_result():
    """Display existing calibration and show undistorted camera view"""
    if not CALIB_JSON.exists():
        sys.exit(f"Run --calibrate first. {CALIB_JSON} not found.")

    r = json.loads(CALIB_JSON.read_text())
    K    = np.array([[r["fx"], 0, r["cx"]], [0, r["fy"], r["cy"]], [0, 0, 1]])
    dist = np.array([r["dist"]["k1"], r["dist"]["k2"],
                     r["dist"]["p1"], r["dist"]["p2"], r["dist"]["k3"]])
    print(json.dumps(r, indent=2))

    cap = cv2.VideoCapture(0)
    print("Showing undistorted feed. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        undist = cv2.undistort(frame, K, dist)
        cv2.imshow("Original",    frame)
        cv2.imshow("Undistorted", undist)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Camera calibration for Archimedes Survey")
    parser.add_argument("--gen-board",  action="store_true", help="Generate checkerboard image")
    parser.add_argument("--capture",    action="store_true", help="Capture calibration images")
    parser.add_argument("--calibrate",  action="store_true", help="Run calibration from images")
    parser.add_argument("--show",       action="store_true", help="Show undistorted feed")
    parser.add_argument("--count",      type=int, default=25, help="Number of images to capture")
    args = parser.parse_args()

    if args.gen_board:
        gen_board()
    elif args.capture:
        capture(args.count)
    elif args.calibrate:
        calibrate()
    elif args.show:
        show_result()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
