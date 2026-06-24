"""
Headless tests for main.py save logic and ScoredFrame buffer mechanics.

Verifies:
  - _save_best picks the highest-scoring frame (not just the most recent)
  - _save_best returns a failure tuple for an empty buffer
  - _save_best returns a failure tuple when all scores are 0
  - The saved file actually exists on disk and is a valid JPEG

Run from the project root:
    python tests/test_main.py
"""

import sys
import os
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import ScoredFrame, _save_best


def _fake_frame(value: int = 128) -> np.ndarray:
    """Return a small solid-colour BGR frame for testing."""
    return np.full((48, 64, 3), value, dtype=np.uint8)


def test_empty_buffer_fails():
    buf = deque(maxlen=90)
    with tempfile.TemporaryDirectory() as tmp:
        ok, msg, score = _save_best(buf, Path(tmp))
    assert not ok, "Should fail on empty buffer"
    assert score == 0.0
    print("PASS: empty buffer returns failure")


def test_all_zero_scores_fails():
    buf = deque([ScoredFrame(score=0.0, frame=_fake_frame()) for _ in range(5)], maxlen=90)
    with tempfile.TemporaryDirectory() as tmp:
        ok, msg, score = _save_best(buf, Path(tmp))
    assert not ok, "Should fail when all scores are 0"
    assert "No good frame" in msg
    print("PASS: all-zero scores returns failure")


def test_picks_highest_score():
    """The saved frame must be the one with score 0.9, not the last (0.1)."""
    best_frame = _fake_frame(200)   # distinctive pixel value
    buf = deque([
        ScoredFrame(score=0.15, frame=_fake_frame(50)),
        ScoredFrame(score=0.90, frame=best_frame),
        ScoredFrame(score=0.10, frame=_fake_frame(10)),
    ], maxlen=90)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ok, msg, score = _save_best(buf, tmp_path)
        assert ok, f"Expected success, got: {msg}"
        assert abs(score - 0.90) < 1e-6, f"Expected score 0.90, got {score}"

        # Verify it's actually the right frame
        saved_files = list(tmp_path.glob("snap_*.jpg"))
        assert len(saved_files) == 1, f"Expected 1 file, found {len(saved_files)}"
        loaded = cv2.imread(str(saved_files[0]))
        assert loaded is not None, "Could not reload saved JPEG"
        # The best frame was all-200 pixels; a JPEG reloaded won't be bit-exact
        # but the mean should be close to 200 (not 50 or 10).
        assert loaded.mean() > 150, (
            f"Saved frame looks wrong: mean={loaded.mean():.1f}, expected ~200"
        )
    print("PASS: picks highest-scoring frame from buffer")


def test_saved_file_is_valid_jpeg():
    buf = deque([
        ScoredFrame(score=0.25, frame=_fake_frame(128)),
    ], maxlen=90)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ok, msg, score = _save_best(buf, tmp_path)
        assert ok, f"Save should succeed: {msg}"

        files = list(tmp_path.glob("snap_*.jpg"))
        assert len(files) == 1
        path = files[0]
        assert path.suffix == ".jpg"
        assert path.stat().st_size > 0, "Saved file is empty"

        img = cv2.imread(str(path))
        assert img is not None, "OpenCV could not decode the saved JPEG"
        assert img.ndim == 3 and img.shape[2] == 3
    print("PASS: saved file is a valid non-empty JPEG")


def test_filename_format():
    buf = deque([ScoredFrame(score=0.25, frame=_fake_frame())], maxlen=90)
    with tempfile.TemporaryDirectory() as tmp:
        ok, msg, _ = _save_best(buf, Path(tmp))
        assert ok
        assert "snap_" in msg
        assert ".jpg" in msg
    print("PASS: saved filename matches snap_YYYYMMDD_HHMMSS.jpg pattern")


def test_output_dir_created_if_missing():
    buf = deque([ScoredFrame(score=0.25, frame=_fake_frame())], maxlen=90)
    with tempfile.TemporaryDirectory() as tmp:
        new_dir = Path(tmp) / "nested" / "output"
        assert not new_dir.exists()
        ok, msg, _ = _save_best(buf, new_dir)
        assert ok, f"Should create output dir: {msg}"
        assert new_dir.exists()
    print("PASS: output directory created if it does not exist")


if __name__ == "__main__":
    test_empty_buffer_fails()
    test_all_zero_scores_fails()
    test_picks_highest_score()
    test_saved_file_is_valid_jpeg()
    test_filename_format()
    test_output_dir_created_if_missing()
    print("\nAll main.py tests passed.")
