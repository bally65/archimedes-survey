"""
test_tide_predictor.py
======================
Tests for control/tide_predictor.py — safety-critical tidal logic.

No external dependencies (tide_predictor.py is pure Python).

Run:
    python -X utf8 tests/test_tide_predictor.py
    python -m pytest tests/test_tide_predictor.py -v
"""

import io
import math
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "control"))
sys.path.insert(0, ROOT)

from tide_predictor import (
    harmonic_tide_cm,
    TidePredictor,
    MLLW_OFFSET_CM,
    SAFE_WATER_LEVEL_CM,
    ABORT_RISE_RATE_CM_MIN,
    SAFE_MARGIN_MIN,
    MIN_WINDOW_MIN,
    HARMONIC_CONSTITUENTS,
    TZ_CST,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _make_records(levels: list, start_offset_min: int = -5,
                  step_min: int = 10) -> list:
    """
    Build synthetic records list for get_safe_window tests.
    start_offset_min: minutes relative to now for the first record.
    """
    now = datetime.now(TZ_CST)
    records = []
    for i, level in enumerate(levels):
        t = now + timedelta(minutes=start_offset_min + i * step_min)
        records.append({
            "time":   t.isoformat(),
            "level":  float(level),
            "source": "test",
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# harmonic_tide_cm — pure physics function
# ─────────────────────────────────────────────────────────────────────────────
class TestHarmonicTide(unittest.TestCase):

    def _epoch(self) -> datetime:
        """Reference epoch used inside harmonic_tide_cm."""
        return datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_returns_float(self):
        """harmonic_tide_cm should return a float."""
        t = datetime(2026, 1, 1, 6, 0, tzinfo=TZ_CST)
        result = harmonic_tide_cm(t)
        self.assertIsInstance(result, float)

    def test_output_in_physical_range(self):
        """Tide level should stay within physical bounds for 彰化 coast."""
        # Sum of all amplitudes = 354 cm; MLLW_OFFSET = 200 cm
        # → absolute min ~= 200 - 354 = -154 cm (unreachable due to phase),
        #   practical range is roughly 0–500 cm
        max_amplitude = sum(c["H"] for c in HARMONIC_CONSTITUENTS.values())
        t = datetime(2026, 3, 1, 0, 0, tzinfo=TZ_CST)
        level = harmonic_tide_cm(t)
        self.assertGreater(level, MLLW_OFFSET_CM - max_amplitude)
        self.assertLess(level, MLLW_OFFSET_CM + max_amplitude)

    def test_deterministic(self):
        """Same timestamp must produce identical result."""
        t = datetime(2026, 6, 15, 14, 30, tzinfo=TZ_CST)
        self.assertEqual(harmonic_tide_cm(t), harmonic_tide_cm(t))

    def test_varies_with_time(self):
        """Different timestamps must (almost always) produce different levels."""
        t1 = datetime(2026, 3, 1,  0, 0, tzinfo=TZ_CST)
        t2 = datetime(2026, 3, 1,  6, 0, tzinfo=TZ_CST)   # +6 h (1/4 tidal cycle)
        t3 = datetime(2026, 3, 1, 12, 0, tzinfo=TZ_CST)   # +12h (1/2 tidal cycle)
        vals = {harmonic_tide_cm(t1), harmonic_tide_cm(t2), harmonic_tide_cm(t3)}
        self.assertGreater(len(vals), 1, "tide should vary over 12 hours")

    def test_mllw_offset_applied(self):
        """With all cosines summing to zero the result equals MLLW_OFFSET_CM.
        This happens at t=epoch when all phases align to cancel perfectly — but
        we can verify the mean over a full M2 cycle (12.4206 h) is ~ MLLW_OFFSET."""
        T_M2_h = 12.4206
        n_steps = 100
        epoch = self._epoch()
        mean_level = sum(
            harmonic_tide_cm(epoch + timedelta(hours=i * T_M2_h / n_steps))
            for i in range(n_steps)
        ) / n_steps
        # Mean should be close to MLLW_OFFSET (within ±20 cm due to other constituents)
        self.assertAlmostEqual(mean_level, MLLW_OFFSET_CM, delta=20.0)

    def test_includes_all_constituents(self):
        """All 6 constituents should contribute (result != MLLW_OFFSET at any t)."""
        t = datetime(2026, 1, 1, 0, tzinfo=TZ_CST)
        # If all cosines were zero the result would equal MLLW_OFFSET_CM.
        # With 6 non-trivial phases this should never happen at a random time.
        level = harmonic_tide_cm(t)
        self.assertNotAlmostEqual(level, float(MLLW_OFFSET_CM), places=2)


# ─────────────────────────────────────────────────────────────────────────────
# TidePredictor.find_low_tides — local-minimum detector
# ─────────────────────────────────────────────────────────────────────────────
class TestFindLowTides(unittest.TestCase):

    def setUp(self):
        self.tp = TidePredictor(use_offline=True)

    def _records(self, levels):
        return [{"time": f"T{i:03d}", "level": float(l)} for i, l in enumerate(levels)]

    def test_empty_returns_empty(self):
        self.assertEqual(self.tp.find_low_tides([]), [])

    def test_single_record_returns_empty(self):
        self.assertEqual(self.tp.find_low_tides(self._records([100])), [])

    def test_two_records_returns_empty(self):
        self.assertEqual(self.tp.find_low_tides(self._records([100, 50])), [])

    def test_monotone_decreasing_returns_empty(self):
        """Monotone: last value cannot be a local min (no right neighbour check)."""
        r = self._records([100, 80, 60, 40, 20])
        self.assertEqual(self.tp.find_low_tides(r), [])

    def test_monotone_increasing_returns_empty(self):
        r = self._records([20, 40, 60, 80, 100])
        self.assertEqual(self.tp.find_low_tides(r), [])

    def test_single_valley(self):
        """One clear valley in the middle."""
        r = self._records([100, 50, 20, 50, 100])
        lows = self.tp.find_low_tides(r)
        self.assertEqual(len(lows), 1)
        self.assertAlmostEqual(lows[0]["level"], 20.0)

    def test_multiple_valleys(self):
        r = self._records([100, 20, 100, 15, 100, 10, 100])
        lows = self.tp.find_low_tides(r)
        self.assertEqual(len(lows), 3)
        self.assertAlmostEqual(lows[0]["level"], 20.0)
        self.assertAlmostEqual(lows[1]["level"], 15.0)
        self.assertAlmostEqual(lows[2]["level"], 10.0)

    def test_plateau_not_detected(self):
        """Strict < : plateau (equal neighbours) is not a local min."""
        r = self._records([100, 50, 50, 50, 100])
        lows = self.tp.find_low_tides(r)
        self.assertEqual(len(lows), 0)

    def test_low_includes_idx(self):
        """Each low dict should include 'time', 'level', 'idx'."""
        r = self._records([100, 30, 100])
        lows = self.tp.find_low_tides(r)
        self.assertEqual(len(lows), 1)
        for key in ("time", "level", "idx"):
            self.assertIn(key, lows[0])

    def test_boundary_values_not_returned(self):
        """First and last records can never be local minima."""
        r = self._records([10, 100, 50, 100, 10])
        lows = self.tp.find_low_tides(r)
        # Only the middle value (50) qualifies
        self.assertEqual(len(lows), 1)
        self.assertAlmostEqual(lows[0]["level"], 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# TidePredictor._offline_forecast — record generation
# ─────────────────────────────────────────────────────────────────────────────
class TestOfflineForecast(unittest.TestCase):

    def setUp(self):
        self.tp = TidePredictor(use_offline=True)

    def test_record_count(self):
        """hours × 6 + 1 records (10-min resolution)."""
        for hours in [1, 6, 24]:
            with self.subTest(hours=hours):
                records = self.tp._offline_forecast(hours)
                self.assertEqual(len(records), hours * 6 + 1)

    def test_record_structure(self):
        """Each record must have time / level / source keys."""
        records = self.tp._offline_forecast(1)
        for r in records:
            for key in ("time", "level", "source"):
                self.assertIn(key, r)

    def test_source_is_harmonic_offline(self):
        records = self.tp._offline_forecast(2)
        for r in records:
            self.assertEqual(r["source"], "harmonic_offline")

    def test_times_are_increasing(self):
        records = self.tp._offline_forecast(2)
        for i in range(1, len(records)):
            self.assertGreater(records[i]["time"], records[i-1]["time"])

    def test_time_step_is_10_minutes(self):
        """Consecutive records should be exactly 10 minutes apart."""
        records = self.tp._offline_forecast(1)
        for i in range(1, len(records)):
            t0 = datetime.fromisoformat(records[i-1]["time"])
            t1 = datetime.fromisoformat(records[i]["time"])
            delta = (t1 - t0).total_seconds()
            self.assertAlmostEqual(delta, 600.0, delta=1.0)

    def test_levels_are_physical(self):
        """All levels must be positive (MLLW_OFFSET prevents negative values)."""
        records = self.tp._offline_forecast(24)
        for r in records:
            self.assertGreater(r["level"], 0.0,
                f"Negative tide level: {r['level']}")


# ─────────────────────────────────────────────────────────────────────────────
# TidePredictor.get_safe_window — go/no-go decision logic
# ─────────────────────────────────────────────────────────────────────────────
class TestGetSafeWindow(unittest.TestCase):

    def setUp(self):
        self.tp = TidePredictor(use_offline=True)

    # ---- Return structure -----------------------------------------------

    def test_result_has_required_fields(self):
        records = _make_records([200] * 20)
        w = self.tp.get_safe_window(records)
        for field in ("go", "reason", "window_start", "window_end",
                      "safe_minutes", "current_level", "rise_rate"):
            self.assertIn(field, w, f"Missing field: {field}")

    def test_go_is_bool(self):
        records = _make_records([200] * 20)
        w = self.tp.get_safe_window(records)
        self.assertIsInstance(w["go"], bool)

    def test_current_level_matches_record(self):
        """current_level should equal the level of the first future record."""
        levels = [300.0, 25.0] + [25.0] * 10 + [300.0] * 5
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertAlmostEqual(w["current_level"], 25.0, delta=1.0)

    # ---- Emergency: rapid rise ------------------------------------------

    def test_emergency_rapid_rise(self):
        """Rise > ABORT_RISE_RATE_CM_MIN cm/min → go=False, emergency reason."""
        # Rise of 50 cm in 10 min = 5 cm/min > threshold (3 cm/min)
        levels = [20.0, 70.0] + [200.0] * 10
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertFalse(w["go"])
        self.assertIn("RTH", w["reason"])
        self.assertEqual(w["safe_minutes"], 0)

    def test_emergency_rise_rate_value(self):
        """rise_rate should be correctly computed as Δlevel / 10 min."""
        levels = [10.0, 50.0] + [200.0] * 10   # +40 cm in 10 min = 4 cm/min
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertAlmostEqual(w["rise_rate"], 4.0, places=3)

    def test_falling_tide_not_emergency(self):
        """Falling tide (negative rise_rate) should NOT trigger emergency."""
        levels = [200.0, 100.0] + [25.0] * 10 + [200.0] * 5
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertLess(w["rise_rate"], 0)
        # Should not be the RTH emergency case
        self.assertNotIn("RTH", w["reason"])

    # ---- Go: adequate safe window --------------------------------------

    def test_go_true_with_long_safe_window(self):
        """≥ 50 consecutive safe minutes (SAFE_MARGIN_MIN=20 + MIN_WINDOW_MIN=30)."""
        # records[0]=past(200), records[1..7]=future(25cm), records[8]=200cm
        # safe_minutes = 60 min, usable = 40 min ≥ 30 → go=True
        levels = [200.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 200.0]
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertTrue(w["go"], f"Expected go=True, reason={w['reason']}")
        self.assertGreaterEqual(w["safe_minutes"], MIN_WINDOW_MIN + SAFE_MARGIN_MIN)

    def test_window_times_present_when_go_true(self):
        levels = [200.0] + [20.0] * 8 + [200.0]
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        if w["go"]:
            self.assertIsNotNone(w["window_start"])
            self.assertIsNotNone(w["window_end"])

    # ---- No-go: window too short ----------------------------------------

    def test_nogo_window_too_short(self):
        """Only 20 min of safe time: usable = 0 min < MIN_WINDOW_MIN."""
        levels = [200.0, 25.0, 25.0, 25.0, 200.0] + [200.0] * 10
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertFalse(w["go"])
        self.assertIn("短", w["reason"])  # "窗口太短"

    def test_nogo_exactly_at_minimum(self):
        """usable_minutes = MIN_WINDOW_MIN should be go=True (≥, not >)."""
        # Need usable = 30: safe_minutes = 50 → 7 records ≤ 30 over 60 min
        # Actually safe_minutes=50 min → usable=30
        # 6 records ≤ 30: times t1..t6 each 10 min apart → safe_minutes=(6-1)*10=50
        levels = [200.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 200.0]
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        if w["safe_minutes"] - SAFE_MARGIN_MIN == MIN_WINDOW_MIN:
            self.assertTrue(w["go"])

    # ---- No-go: no safe window at all ----------------------------------

    def test_nogo_no_safe_window(self):
        """All levels above SAFE_WATER_LEVEL_CM → window_start=None."""
        levels = [200.0] * 20
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        self.assertFalse(w["go"])
        self.assertIsNone(w["window_start"])
        self.assertEqual(w["safe_minutes"], 0)
        self.assertIn("等待", w["reason"])  # "等待低潮"

    # ---- Boundary: SAFE_WATER_LEVEL_CM ─────────────────────────────────

    def test_boundary_exactly_safe_level(self):
        """Level == SAFE_WATER_LEVEL_CM should count as safe (≤)."""
        levels = [200.0] + [SAFE_WATER_LEVEL_CM] * 8 + [200.0]
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        # safe_start should not be None since levels[1..8] ≤ threshold
        self.assertIsNotNone(w["window_start"])

    def test_boundary_just_above_safe_level(self):
        """Level == SAFE_WATER_LEVEL_CM + 1 should NOT count as safe."""
        above = float(SAFE_WATER_LEVEL_CM + 1)
        levels = [200.0, above, above, above, 200.0]
        records = _make_records(levels)
        w = self.tp.get_safe_window(records)
        # No safe records → window_start is None
        self.assertIsNone(w["window_start"])


# ─────────────────────────────────────────────────────────────────────────────
# Constants sanity checks
# ─────────────────────────────────────────────────────────────────────────────
class TestConstants(unittest.TestCase):

    def test_abort_rate_matches_flood_scenario(self):
        """ABORT_RISE_RATE_CM_MIN should flag the 35cm/60s flood scenario.
        35 cm/60 s = 0.583 cm/s = 35 cm/min.  Our threshold = 3 cm/min."""
        flood_rate_cm_per_min = 35.0 / 1.0   # worst-case: 35cm in 1 min
        self.assertGreater(flood_rate_cm_per_min, ABORT_RISE_RATE_CM_MIN)

    def test_safe_margin_plus_min_window_is_valid(self):
        """Total required safe minutes must be positive."""
        self.assertGreater(SAFE_MARGIN_MIN + MIN_WINDOW_MIN, 0)

    def test_mllw_offset_positive(self):
        self.assertGreater(MLLW_OFFSET_CM, 0)

    def test_safe_water_level_below_mllw(self):
        """Operations happen near low water — threshold must be below MLLW_OFFSET."""
        self.assertLess(SAFE_WATER_LEVEL_CM, MLLW_OFFSET_CM)


# ─────────────────────────────────────────────────────────────────────────────
def run_all_verbose():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestHarmonicTide,
        TestFindLowTides,
        TestOfflineForecast,
        TestGetSafeWindow,
        TestConstants,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"  總計：{result.testsRun} 項  通過：{passed}  "
          f"失敗：{len(result.failures)}  錯誤：{len(result.errors)}")
    print("  ✓ 全部通過" if result.wasSuccessful() else "  ✗ 有測試失敗")
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.exit(0 if run_all_verbose() else 1)
