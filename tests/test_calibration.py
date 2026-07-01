"""
Unit tests for src/calibration.py.  Headless — no webcam, no cv2 window.

Run from the project root:
    python tests/test_calibration.py
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calibration import (
    save_calibration,
    load_calibration,
    calibration_blink_threshold,
    compute_calibration_from_samples,
)
from src.config import BLINK_THRESHOLD


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_round_trip():
    data = {"baseline_ear": 0.280, "blink_threshold": 0.224, "sample_count": 120}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calib.json"
        save_calibration(path, data)
        loaded = load_calibration(path)
    assert loaded is not None
    assert math.isclose(loaded["blink_threshold"], 0.224, abs_tol=1e-6)
    assert loaded["sample_count"] == 120
    print("PASS: save_calibration / load_calibration round-trip")


def test_load_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        result = load_calibration(Path(tmp) / "nonexistent.json")
    assert result is None
    print("PASS: load_calibration returns None for missing file")


def test_load_corrupt_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.json"
        p.write_text("not valid json", encoding="utf-8")
        result = load_calibration(p)
    assert result is None
    print("PASS: load_calibration returns None for corrupt file")


# ---------------------------------------------------------------------------
# calibration_blink_threshold
# ---------------------------------------------------------------------------

def test_threshold_from_data():
    data = {"baseline_ear": 0.300, "blink_threshold": 0.240, "sample_count": 100}
    thr = calibration_blink_threshold(data)
    assert math.isclose(thr, 0.240, abs_tol=1e-6)
    print("PASS: calibration_blink_threshold extracts correct value")


def test_threshold_fallback_to_config():
    thr = calibration_blink_threshold({})   # missing key
    assert math.isclose(thr, BLINK_THRESHOLD, abs_tol=1e-6)
    print(f"PASS: calibration_blink_threshold falls back to BLINK_THRESHOLD={BLINK_THRESHOLD}")


# ---------------------------------------------------------------------------
# compute_calibration_from_samples
# ---------------------------------------------------------------------------

def test_compute_from_samples_normal():
    # 100 samples all at 0.30 -> baseline ~0.30, threshold ~0.24
    samples = [0.30] * 100
    data = compute_calibration_from_samples(samples)
    assert math.isclose(data["baseline_ear"], 0.30, abs_tol=0.01)
    assert data["blink_threshold"] < data["baseline_ear"]
    assert data["sample_count"] == 100
    print(f"PASS: compute_calibration_from_samples — baseline={data['baseline_ear']}, "
          f"threshold={data['blink_threshold']}")


def test_compute_from_samples_too_few_returns_default():
    data = compute_calibration_from_samples([0.28, 0.29])  # < _MIN_SAMPLES
    assert math.isclose(data["blink_threshold"], BLINK_THRESHOLD, abs_tol=1e-6)
    assert data["sample_count"] == 2
    print("PASS: compute_calibration_from_samples falls back to default on < 5 samples")


def test_compute_drops_low_outliers():
    # 90 good samples at 0.30, 10 near-zero outliers (blinks during collection)
    samples = [0.30] * 90 + [0.05] * 10
    data = compute_calibration_from_samples(samples)
    # Baseline should be near 0.30, not dragged down by the 10 outliers
    assert data["baseline_ear"] > 0.25, (
        f"Expected baseline > 0.25, got {data['baseline_ear']}"
    )
    print(f"PASS: compute_calibration_from_samples filters low outliers — "
          f"baseline={data['baseline_ear']:.4f}")


if __name__ == "__main__":
    test_save_and_load_round_trip()
    test_load_missing_returns_none()
    test_load_corrupt_returns_none()
    test_threshold_from_data()
    test_threshold_fallback_to_config()
    test_compute_from_samples_normal()
    test_compute_from_samples_too_few_returns_default()
    test_compute_drops_low_outliers()
    print("\nAll calibration tests passed.")
