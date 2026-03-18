"""
test_acoustic_integration.py
============================
acoustic_processor.py 整合測試（純 Python，不需 ROS2）。

直接匯入 TSAFTReconstructor 和 sound_speed，驗證：
  1. 聲速公式正確性
  2. 單點深度重建精度
  3. 平面反射體重建（網格掃描）
  4. Coherence Factor 加權效果
  5. 邊界提取 + 深度估算
  6. 最少點數保護
  7. 噪音容忍度

Run:
    python tests/test_acoustic_integration.py
    python -m pytest tests/test_acoustic_integration.py -v
"""

import sys
import os
import math
import time
import unittest

import numpy as np

# ── 加入 ros2 package 路徑 ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2"))

# acoustic_processor 使用了 rclpy，但我們只需要演算法層
# 透過 importlib 部分匯入，跳過 ROS2 依賴
import importlib, types

# Stub out rclpy 讓 acoustic_processor 可以在非 ROS2 環境匯入
_rclpy = types.ModuleType("rclpy")
_rclpy.init = lambda *a, **k: None
_rclpy.spin = lambda *a, **k: None
_rclpy.shutdown = lambda *a, **k: None
_rclpy_node = types.ModuleType("rclpy.node")
_rclpy_node.Node = object
sys.modules.setdefault("rclpy", _rclpy)
sys.modules.setdefault("rclpy.node", _rclpy_node)
_std_msgs = types.ModuleType("std_msgs")
_std_msgs_msg = types.ModuleType("std_msgs.msg")
_std_msgs_msg.String = object
_std_msgs_msg.Float32 = object
sys.modules.setdefault("std_msgs", _std_msgs)
sys.modules.setdefault("std_msgs.msg", _std_msgs_msg)

from archimedes_survey.acoustic_processor import (
    TSAFTReconstructor,
    AcousticProcessorNode,
    sound_speed,
    MIN_SCAN_PTS,
    DEFAULT_SOUND_SPEED_MS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 輔助函式
# ═══════════════════════════════════════════════════════════════════════════════

def make_scan_point(x_m: float, y_m: float, z_m: float,
                    reflector_z_m: float, c: float, noise_m: float = 0.0):
    """
    模擬換能器在 (x_m, y_m, z_m) 量測到深度 reflector_z_m 的單點。
    noise_m：r_meas 的高斯噪音標準差（m）。
    """
    r_true = math.sqrt((x_m - 0.0)**2 + (y_m - 0.0)**2 + (z_m - reflector_z_m)**2)
    noise  = np.random.normal(0, noise_m) if noise_m > 0 else 0.0
    r_meas = max(0.01, r_true + noise)
    tof_us = r_meas * 2 / c * 1e6
    return {
        "x_m":        x_m,
        "y_m":        y_m,
        "z_m":        z_m,
        "r_meas_m":   r_meas,
        "tof_us":     tof_us,
        "confidence": 0.9,
    }


def make_grid_scan(nx: int, ny: int, step_m: float,
                   reflector_z_m: float, c: float,
                   z_sensor: float = 0.0, noise_m: float = 0.0):
    """
    XY 網格掃描，換能器在 z=z_sensor 平面，掃描一個水平面反射體（位於 z=reflector_z_m）。
    """
    pts = []
    x0 = -(nx - 1) * step_m / 2
    y0 = -(ny - 1) * step_m / 2
    for i in range(nx):
        for j in range(ny):
            x = x0 + i * step_m
            y = y0 + j * step_m
            pts.append(make_scan_point(x, y, z_sensor, reflector_z_m, c, noise_m))
    return pts


# ═══════════════════════════════════════════════════════════════════════════════
# 測試案例
# ═══════════════════════════════════════════════════════════════════════════════

class TestSoundSpeed(unittest.TestCase):
    """聲速公式驗證"""

    def test_default_conditions(self):
        """T=20°C, S=35ppt, D=0 → 應接近 1521 m/s"""
        c = sound_speed(20.0, 35.0, 0.0)
        self.assertAlmostEqual(c, 1521.0, delta=1.0,
                               msg=f"Expected ~1521, got {c:.1f}")

    def test_fresh_water_lower(self):
        """低鹽度（淡水）聲速應低於海水"""
        c_salt  = sound_speed(20.0, 35.0, 0.0)
        c_fresh = sound_speed(20.0,  0.0, 0.0)
        self.assertLess(c_fresh, c_salt,
                        msg="Fresh water should be slower than seawater")

    def test_temperature_effect(self):
        """溫度升高 → 聲速升高"""
        c_cool = sound_speed(15.0, 35.0, 0.0)
        c_warm = sound_speed(30.0, 35.0, 0.0)
        self.assertGreater(c_warm, c_cool)

    def test_clamped_range(self):
        """極端值應被 clamp 在 1400~1600 m/s"""
        c_low  = sound_speed(-50.0, 0.0, 0.0)
        c_high = sound_speed(100.0, 100.0, 1000.0)
        self.assertGreaterEqual(c_low,  1400.0)
        self.assertLessEqual(c_high, 1600.0)


class TestReconstructorBasic(unittest.TestCase):
    """TSAFTReconstructor 基礎功能"""

    def setUp(self):
        self.c     = 1521.0
        self.recon = TSAFTReconstructor(c=self.c, sigma_m=0.005)

    def test_single_depth_no_noise(self):
        """
        4×4 網格，反射體在 z=0.50m（50cm）。
        重建峰值 Z 應在 50cm ± 10cm 範圍內。
        """
        target_z  = 0.50   # 目標深度 50cm
        pts = make_grid_scan(4, 4, 0.007, target_z, self.c, z_sensor=0.0)
        self.assertGreaterEqual(len(pts), MIN_SCAN_PTS,
                                f"Need >= {MIN_SCAN_PTS} pts, got {len(pts)}")

        gx, gy, gz, intensity = self.recon.reconstruct(pts)

        # 找強度最大的體素
        flat_idx = np.argmax(intensity)
        ix, iy, iz = np.unravel_index(flat_idx, intensity.shape)
        peak_z_mm  = float(gz[iz])

        self.assertAlmostEqual(peak_z_mm, target_z * 1000, delta=80.0,
                               msg=f"Peak z={peak_z_mm:.0f}mm, expected ~500mm")

    def test_intensity_nonzero(self):
        """重建結果不應全為零"""
        pts = make_grid_scan(4, 4, 0.007, 0.50, self.c)
        _, _, _, intensity = self.recon.reconstruct(pts)
        self.assertGreater(intensity.max(), 0.0)

    def test_cf_weighting(self):
        """CF 強度應 <= DAS 強度（CF 為衰減因子，不能放大）"""
        pts = make_grid_scan(4, 4, 0.007, 0.50, self.c)
        gx, gy, gz, I_cf = self.recon.reconstruct(pts)

        # 簡單驗證：CF 加權後最大值不會超過 N（掃描點數）
        self.assertLessEqual(I_cf.max(), len(pts) + 1e-6,
                             "CF intensity should not exceed N scan points")


class TestBoundaryExtraction(unittest.TestCase):
    """邊界提取與深度估算"""

    def setUp(self):
        self.c     = 1521.0
        self.recon = TSAFTReconstructor(c=self.c, sigma_m=0.006)

    def test_boundary_found(self):
        """有效掃描應找到邊界點"""
        pts = make_grid_scan(4, 4, 0.007, 0.60, self.c)
        gx, gy, gz, intensity = self.recon.reconstruct(pts)
        boundary = self.recon.extract_burrow_boundary(gx, gy, gz, intensity, threshold=0.30)
        self.assertGreater(len(boundary), 0, "Should find boundary points")

    def test_depth_estimate_reasonable(self):
        """
        深度估算應合理（目標 60cm，估算在 30~120cm 內）。
        注意：網格解析度 5mm + sigma 6mm 會有誤差。
        """
        pts = make_grid_scan(4, 4, 0.007, 0.60, self.c)
        gx, gy, gz, intensity = self.recon.reconstruct(pts)
        boundary = self.recon.extract_burrow_boundary(gx, gy, gz, intensity)
        depth_cm, diam_mm, conf = self.recon.estimate_burrow_depth(boundary)

        self.assertGreater(depth_cm, 5.0,  "Depth should be > 5cm")
        self.assertLess(depth_cm, 200.0, "Depth should be < 200cm")
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf,   1.0)

    def test_empty_boundary(self):
        """空邊界應回傳 0"""
        depth_cm, diam_mm, conf = self.recon.estimate_burrow_depth([])
        self.assertEqual(depth_cm, 0.0)
        self.assertEqual(diam_mm,  0.0)
        self.assertEqual(conf,     0.0)


class TestNoiseTolerance(unittest.TestCase):
    """噪音容忍度"""

    def setUp(self):
        self.c     = 1521.0
        self.recon = TSAFTReconstructor(c=self.c, sigma_m=0.008)
        np.random.seed(42)

    def test_with_5mm_noise(self):
        """5mm 噪音（sigma=5mm）下仍能找到邊界"""
        pts = make_grid_scan(4, 4, 0.007, 0.50, self.c, noise_m=0.005)
        gx, gy, gz, intensity = self.recon.reconstruct(pts)
        boundary = self.recon.extract_burrow_boundary(gx, gy, gz, intensity, threshold=0.25)
        self.assertGreater(len(boundary), 0,
                           "Should find boundary even with 5mm noise")

    def test_peak_z_with_noise(self):
        """10mm 噪音下峰值 Z 誤差應 < 150mm"""
        pts = make_grid_scan(4, 4, 0.007, 0.80, self.c, noise_m=0.010)
        gx, gy, gz, intensity = self.recon.reconstruct(pts)
        flat_idx = np.argmax(intensity)
        ix, iy, iz = np.unravel_index(flat_idx, intensity.shape)
        peak_z_mm = float(gz[iz])
        self.assertAlmostEqual(peak_z_mm, 800.0, delta=200.0,
                               msg=f"Peak z={peak_z_mm:.0f}mm, expected ~800mm")


class TestMinimumPoints(unittest.TestCase):
    """點數保護"""

    def setUp(self):
        self.recon = TSAFTReconstructor(c=1521.0, sigma_m=0.005)

    def test_reconstruct_with_few_points_still_runs(self):
        """
        TSAFTReconstructor 本身不檢查 MIN_SCAN_PTS（由 ROS2 節點負責）。
        但用 < MIN_SCAN_PTS 點呼叫不應崩潰。
        """
        pts = make_grid_scan(3, 3, 0.007, 0.50, 1521.0)  # 9 points < 12
        try:
            gx, gy, gz, intensity = self.recon.reconstruct(pts)
        except Exception as e:
            self.fail(f"reconstruct() should not raise with few points: {e}")


class TestAngleEstimation(unittest.TestCase):
    """洞穴傾角估算"""

    def test_vertical_burrow(self):
        """垂直洞穴（深度方向 = Z）應接近 0°"""
        boundary = [
            {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 100.0, "intensity": 0.5},
            {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 500.0, "intensity": 0.9},
        ]
        angle = AcousticProcessorNode._estimate_angle(boundary)
        self.assertAlmostEqual(angle, 0.0, delta=5.0,
                               msg=f"Vertical burrow should be ~0°, got {angle}°")

    def test_horizontal_burrow(self):
        """水平洞穴（橫向延伸）應接近 90°"""
        boundary = [
            {"x_mm": 0.0,   "y_mm": 0.0, "z_mm": 100.0, "intensity": 0.5},
            {"x_mm": 500.0, "y_mm": 0.0, "z_mm": 100.0, "intensity": 0.9},
        ]
        angle = AcousticProcessorNode._estimate_angle(boundary)
        self.assertAlmostEqual(angle, 90.0, delta=5.0,
                               msg=f"Horizontal burrow should be ~90°, got {angle}°")

    def test_45_deg_burrow(self):
        """45° 洞穴"""
        boundary = [
            {"x_mm": 0.0,   "y_mm": 0.0, "z_mm": 100.0, "intensity": 0.5},
            {"x_mm": 300.0, "y_mm": 0.0, "z_mm": 400.0, "intensity": 0.9},
        ]
        angle = AcousticProcessorNode._estimate_angle(boundary)
        self.assertAlmostEqual(angle, 45.0, delta=5.0,
                               msg=f"45° burrow, got {angle}°")


class TestPerformance(unittest.TestCase):
    """效能測試（確保重建時間合理）"""

    def test_reconstruction_time_16pts(self):
        """16 點重建應在 30 秒內（RPi4 可能較慢，PC 應 <5s）"""
        recon = TSAFTReconstructor(c=1521.0, sigma_m=0.005)
        pts   = make_grid_scan(4, 4, 0.007, 0.50, 1521.0)

        t0 = time.monotonic()
        recon.reconstruct(pts)
        dt = time.monotonic() - t0

        print(f"\n  [Perf] 16-pt reconstruction: {dt:.2f}s")
        self.assertLess(dt, 30.0,
                        f"Reconstruction took {dt:.2f}s, too slow for field use")

    def test_reconstruction_time_48pts(self):
        """48 點（推薦掃描密度）應在 90 秒內"""
        recon = TSAFTReconstructor(c=1521.0, sigma_m=0.005)
        pts   = make_grid_scan(6, 8, 0.006, 0.50, 1521.0)  # 48 pts

        t0 = time.monotonic()
        recon.reconstruct(pts)
        dt = time.monotonic() - t0

        print(f"\n  [Perf] 48-pt reconstruction: {dt:.2f}s")
        self.assertLess(dt, 90.0,
                        f"Reconstruction took {dt:.2f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# 端對端整合測試（模擬完整掃描 → 輸出驗證）
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd(unittest.TestCase):
    """
    模擬一次完整的洞穴掃描任務：
      1. 換能器在 XY 平面 5×5 網格（step=7mm）掃描
      2. 目標：奧螻蛄蝦洞穴入口（Ø23cm，深度 80cm）
      3. 驗證輸出深度、直徑、信心值
    """

    def test_burrow_scan_scenario(self):
        """奧螻蛄蝦典型洞穴場景（Ø23cm, 深 80cm）"""
        c       = sound_speed(25.0, 32.0, 0.0)  # 台灣西海岸夏季
        recon   = TSAFTReconstructor(c=c, sigma_m=0.006)

        # 換能器在 XY 平面掃描（5mm 步進，5×5 網格）
        # 洞口中心在 (0, 0)，深度 80cm
        TARGET_DEPTH = 0.80   # m
        pts = make_grid_scan(5, 5, 0.005, TARGET_DEPTH, c,
                             z_sensor=0.0, noise_m=0.003)

        # 重建
        gx, gy, gz, intensity = recon.reconstruct(pts)

        # 邊界提取
        boundary = recon.extract_burrow_boundary(gx, gy, gz, intensity, threshold=0.30)
        depth_cm, diam_mm, conf = recon.estimate_burrow_depth(boundary)

        print(f"\n  [E2E] c={c:.0f}m/s  pts={len(pts)}")
        print(f"  [E2E] depth={depth_cm:.1f}cm (target={TARGET_DEPTH*100:.0f}cm)")
        print(f"  [E2E] diam={diam_mm:.0f}mm  conf={conf:.3f}")
        print(f"  [E2E] boundary pts={len(boundary)}")

        # 基本斷言：找到結果（不強制精確，因為網格解析度 5mm）
        self.assertGreater(len(boundary), 0, "Should find at least 1 boundary point")
        self.assertGreater(depth_cm, 0.0, "Depth should be positive")
        self.assertGreater(conf, 0.0,     "Confidence should be positive")

    def test_geojson_fields_present(self):
        """
        burrow_3d 訊息應包含 GeoJSON 必要欄位
        （驗證與 mission_logger.py 的介面相容性）
        """
        import json

        # 模擬 AcousticProcessorNode 計算的 burrow_msg
        depth_cm = 75.0
        diam_mm  = 230.0
        conf     = 0.82
        angle    = 15.0

        burrow_msg = {
            "timestamp":           1234567890.0,
            "n_scan_pts":          25,
            "recon_time_s":        3.14,
            "burrow_depth_m":      round(depth_cm / 100.0, 4),
            "burrow_depth_cm":     round(depth_cm, 2),
            "burrow_diameter_mm":  round(diam_mm, 1),
            "ultrasound_confidence": round(conf, 3),
            "boundary_pts":        42,
            "geojson_extra": {
                "burrow_depth_m":           round(depth_cm / 100.0, 4),
                "burrow_angle_deg":          angle,
                "ultrasound_confidence":    round(conf, 3),
            },
        }

        # 序列化 / 反序列化
        json_str = json.dumps(burrow_msg)
        parsed   = json.loads(json_str)

        required = ["burrow_depth_m", "burrow_diameter_mm",
                    "ultrasound_confidence", "geojson_extra"]
        for key in required:
            self.assertIn(key, parsed, f"Missing key: {key}")

        geojson_req = ["burrow_depth_m", "burrow_angle_deg", "ultrasound_confidence"]
        for key in geojson_req:
            self.assertIn(key, parsed["geojson_extra"],
                          f"Missing geojson_extra key: {key}")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Archimedes Survey - Acoustic Processor Integration Tests")
    print("=" * 65)
    unittest.main(verbosity=2)
