"""
Unit tests for src/feedback.py (Feature 3 — adaptive learning).

All tests are headless and do not touch the webcam or GUI.

Run from the project root:
    python tests/test_feedback.py
"""

import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feedback import log_decision, decision_count, compute_adjusted_threshold
from src.config import AUTO_SNAP_THRESHOLD


# ---------------------------------------------------------------------------
# log_decision + decision_count
# ---------------------------------------------------------------------------

def test_log_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        assert not path.exists()
        log_decision(path, score=0.30, kept=True)
        assert path.exists()
    print("PASS: log_decision creates file if absent")


def test_log_appends_entries():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        log_decision(path, score=0.30, kept=True)
        log_decision(path, score=0.25, kept=False)
        log_decision(path, score=0.28, kept=True)
        assert decision_count(path) == 3
    print("PASS: log_decision appends; decision_count returns 3")


def test_log_fields():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        log_decision(path, score=0.312, kept=True)
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data[0]
        assert "timestamp" in entry
        assert math.isclose(entry["score"], 0.312, abs_tol=1e-4)
        assert entry["kept"] is True
    print("PASS: log_decision stores timestamp, score, kept")


def test_decision_count_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        n = decision_count(Path(tmp) / "nonexistent.json")
    assert n == 0
    print("PASS: decision_count returns 0 for missing file")


# ---------------------------------------------------------------------------
# compute_adjusted_threshold
# ---------------------------------------------------------------------------

def test_no_decisions_returns_base():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert math.isclose(thr, 0.26, abs_tol=1e-6)
    print("PASS: compute_adjusted_threshold — no file -> base unchanged")


def test_below_window_no_adjustment():
    """19 decisions (< window 20) should NOT trigger an adjustment."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        for _ in range(19):
            log_decision(path, score=0.20, kept=False)   # all discards
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert math.isclose(thr, 0.26, abs_tol=1e-6), (
        f"Expected unchanged 0.26, got {thr}"
    )
    print("PASS: compute_adjusted_threshold — 19 decisions -> no adjustment")


def test_all_discards_raises_threshold():
    """20 decisions all discarded -> threshold raised by step."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        for _ in range(20):
            log_decision(path, score=0.20, kept=False)
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert math.isclose(thr, 0.27, abs_tol=1e-6), (
        f"Expected 0.27 (raised by step), got {thr}"
    )
    print(f"PASS: compute_adjusted_threshold — all discards -> raised to {thr:.3f}")


def test_all_kept_lowers_threshold():
    """20 decisions all kept -> threshold lowered by step."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        for _ in range(20):
            log_decision(path, score=0.30, kept=True)
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert math.isclose(thr, 0.25, abs_tol=1e-6), (
        f"Expected 0.25 (lowered by step), got {thr}"
    )
    print(f"PASS: compute_adjusted_threshold — all kept -> lowered to {thr:.3f}")


def test_neutral_rate_no_change():
    """14/20 kept (70%) is in the neutral band [50%, 80%] -> no change."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        for i in range(20):
            log_decision(path, score=0.28, kept=(i < 14))  # 70% kept
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert math.isclose(thr, 0.26, abs_tol=1e-6), (
        f"Expected unchanged 0.26 (neutral rate), got {thr}"
    )
    print("PASS: compute_adjusted_threshold — 70% kept -> no adjustment (neutral band)")


def test_threshold_clamped_to_max():
    """Repeated raises must not exceed 0.45."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        for _ in range(200):   # 10 windows of 20, all discards
            log_decision(path, score=0.10, kept=False)
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert thr <= 0.45, f"Threshold should be clamped at 0.45, got {thr}"
    print(f"PASS: compute_adjusted_threshold — repeated raises clamped to {thr:.3f}")


def test_threshold_clamped_to_min():
    """Repeated lowers must not go below 0.15."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        for _ in range(200):   # 10 windows of 20, all kept
            log_decision(path, score=0.35, kept=True)
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert thr >= 0.15, f"Threshold should be clamped at 0.15, got {thr}"
    print(f"PASS: compute_adjusted_threshold — repeated lowers clamped to {thr:.3f}")


def test_two_complete_windows():
    """Two complete windows with opposite signals should partially cancel."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "feedback.json"
        # Window 1: all discards -> +0.01
        for _ in range(20):
            log_decision(path, score=0.10, kept=False)
        # Window 2: all kept -> -0.01 (back to base)
        for _ in range(20):
            log_decision(path, score=0.35, kept=True)
        thr = compute_adjusted_threshold(path, base_threshold=0.26, window=20, step=0.01)
    assert math.isclose(thr, 0.26, abs_tol=1e-6), (
        f"Expected 0.26 after opposite signals, got {thr}"
    )
    print("PASS: compute_adjusted_threshold — two opposite windows cancel out")


if __name__ == "__main__":
    test_log_creates_file()
    test_log_appends_entries()
    test_log_fields()
    test_decision_count_missing_file()
    test_no_decisions_returns_base()
    test_below_window_no_adjustment()
    test_all_discards_raises_threshold()
    test_all_kept_lowers_threshold()
    test_neutral_rate_no_change()
    test_threshold_clamped_to_max()
    test_threshold_clamped_to_min()
    test_two_complete_windows()
    print("\nAll feedback / adaptive-learning tests passed.")
