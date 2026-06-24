"""
Unit tests for scoring.py.  All tests are headless (no webcam, no MediaPipe).

Run from the project root:
    python tests/test_scoring.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scoring import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
    is_blinking,
    face_eye_score,
    frame_score,
    score_faces,
    BLINK_THRESHOLD,
    FaceScore,
    FrameScore,
)
from src.config import ENABLE_MOUTH_PENALTY


# ---------------------------------------------------------------------------
# Helpers to build synthetic FaceResult-like objects
# ---------------------------------------------------------------------------

class _FakeEye:
    """Synthetic eye point array for testing without a real FaceResult."""
    def __init__(self, pts):
        self.pts = np.array(pts, dtype=np.float32)


def _make_eye_pts(ear_target: float, width: float = 100.0) -> np.ndarray:
    """Build a (6, 2) array whose EAR equals ear_target.

    Layout (all on a single horizontal axis except the lid points):
      P1=(0, 0), P4=(width, 0)  -> horizontal = width
      P2=(w/3, v), P6=(w/3, -v)
      P3=(2w/3, v), P5=(2w/3, -v)
    where v is chosen so EAR = (2v + 2v) / (2 * width) = ear_target
      => v = ear_target * width / 2
    """
    v = ear_target * width / 2.0
    w = width
    return np.array([
        [0,     0],   # P1 outer corner
        [w/3,   v],   # P2 upper-outer
        [2*w/3, v],   # P3 upper-inner
        [w,     0],   # P4 inner corner
        [2*w/3, -v],  # P5 lower-inner
        [w/3,   -v],  # P6 lower-outer
    ], dtype=np.float32)


class _FakeFace:
    """Minimal duck-type for FaceResult, usable by scoring functions."""
    def __init__(self, face_index: int, right_ear: float, left_ear: float,
                 mar: float = 0.0):
        self.face_index    = face_index
        self.right_eye_pts = _make_eye_pts(right_ear)
        self.left_eye_pts  = _make_eye_pts(left_ear)
        # Build a synthetic mouth with the requested MAR.
        # MAR = vertical / horizontal; we fix horizontal = 100, vary vertical.
        self.mouth_pts = np.array([
            [50.0,    0.0],          # upper_inner
            [50.0, mar * 100.0],     # lower_inner  (vertical = mar * 100)
            [ 0.0, mar * 50.0],      # left_corner
            [100.0, mar * 50.0],     # right_corner (horizontal = 100)
        ], dtype=np.float32)


# ---------------------------------------------------------------------------
# EAR tests
# ---------------------------------------------------------------------------

def test_ear_wide_open():
    pts = _make_eye_pts(0.30)
    ear = eye_aspect_ratio(pts)
    assert math.isclose(ear, 0.30, abs_tol=1e-5), f"Expected ~0.30, got {ear}"
    print("PASS: EAR wide-open (0.30)")


def test_ear_blinking():
    pts = _make_eye_pts(0.10)
    ear = eye_aspect_ratio(pts)
    assert math.isclose(ear, 0.10, abs_tol=1e-5), f"Expected ~0.10, got {ear}"
    print("PASS: EAR blinking (0.10)")


def test_ear_fully_closed():
    # Vertical distance -> 0; eye fully shut
    pts = _make_eye_pts(0.0)
    ear = eye_aspect_ratio(pts)
    assert math.isclose(ear, 0.0, abs_tol=1e-5), f"Expected 0.0, got {ear}"
    print("PASS: EAR fully closed (0.0)")


def test_ear_degenerate_horizontal():
    # All points at the origin -> horizontal = 0 -> should return 0.0, not raise
    pts = np.zeros((6, 2), dtype=np.float32)
    ear = eye_aspect_ratio(pts)
    assert ear == 0.0, f"Expected 0.0 for degenerate pts, got {ear}"
    print("PASS: EAR degenerate (all-zeros -> 0.0)")


# ---------------------------------------------------------------------------
# is_blinking tests
# ---------------------------------------------------------------------------

def test_is_blinking_above_threshold():
    assert not is_blinking(BLINK_THRESHOLD + 0.05)
    print(f"PASS: is_blinking(>{BLINK_THRESHOLD}) -> False")


def test_is_blinking_below_threshold():
    assert is_blinking(BLINK_THRESHOLD - 0.05)
    print(f"PASS: is_blinking(<{BLINK_THRESHOLD}) -> True")


def test_is_blinking_exactly_at_threshold():
    # Strictly less-than, so exactly at threshold is NOT blinking
    assert not is_blinking(BLINK_THRESHOLD)
    print(f"PASS: is_blinking(=={BLINK_THRESHOLD}) -> False (strict <)")


# ---------------------------------------------------------------------------
# face_eye_score tests
# ---------------------------------------------------------------------------

def test_face_eye_score_average():
    face = _FakeFace(face_index=0, right_ear=0.30, left_ear=0.20)
    score = face_eye_score(face)
    assert math.isclose(score, 0.25, abs_tol=1e-5), f"Expected 0.25, got {score}"
    print("PASS: face_eye_score averages both eyes")


def test_face_eye_score_symmetric():
    face = _FakeFace(face_index=0, right_ear=0.25, left_ear=0.25)
    assert math.isclose(face_eye_score(face), 0.25, abs_tol=1e-5)
    print("PASS: face_eye_score symmetric case")


# ---------------------------------------------------------------------------
# frame_score tests
# ---------------------------------------------------------------------------

def test_frame_score_empty_faces():
    assert frame_score([]) == 0.0
    print("PASS: frame_score([]) -> 0.0")


def test_frame_score_single_face():
    face = _FakeFace(face_index=0, right_ear=0.28, left_ear=0.28)
    score = frame_score([face])
    assert math.isclose(score, 0.28, abs_tol=1e-5), f"Expected 0.28, got {score}"
    print("PASS: frame_score single face")


def test_frame_score_takes_minimum():
    face_good  = _FakeFace(face_index=0, right_ear=0.30, left_ear=0.30)
    face_blink = _FakeFace(face_index=1, right_ear=0.10, left_ear=0.10)
    score = frame_score([face_good, face_blink])
    # Must equal the worse face's score
    assert math.isclose(score, 0.10, abs_tol=1e-5), f"Expected 0.10, got {score}"
    print("PASS: frame_score takes minimum (blinker tanks the frame)")


def test_frame_score_three_faces_one_blinks():
    faces = [
        _FakeFace(0, right_ear=0.30, left_ear=0.30),
        _FakeFace(1, right_ear=0.27, left_ear=0.29),
        _FakeFace(2, right_ear=0.08, left_ear=0.09),  # blink
    ]
    score = frame_score(faces)
    assert score < BLINK_THRESHOLD, f"Blinker should tank score, got {score}"
    print("PASS: frame_score 3 faces, one blinks -> score < threshold")


# ---------------------------------------------------------------------------
# score_faces tests (FrameScore / FaceScore)
# ---------------------------------------------------------------------------

def test_score_faces_empty():
    result = score_faces([])
    assert isinstance(result, FrameScore)
    assert result.score == 0.0
    assert not result.has_blinker
    assert result.face_scores == []
    print("PASS: score_faces([]) -> FrameScore(score=0.0, has_blinker=False)")


def test_score_faces_all_open():
    faces = [_FakeFace(0, 0.28, 0.30), _FakeFace(1, 0.26, 0.29)]
    result = score_faces(faces)
    assert result.score >= BLINK_THRESHOLD
    assert not result.has_blinker
    assert len(result.face_scores) == 2
    assert all(isinstance(fs, FaceScore) for fs in result.face_scores)
    print("PASS: score_faces — all open, has_blinker=False")


def test_score_faces_one_blinker():
    faces = [_FakeFace(0, 0.29, 0.28), _FakeFace(1, 0.08, 0.09)]
    result = score_faces(faces)
    assert result.score < BLINK_THRESHOLD
    assert result.has_blinker
    blinkers = [fs for fs in result.face_scores if fs.blinking]
    assert len(blinkers) == 1 and blinkers[0].face_index == 1
    print("PASS: score_faces — one blinker detected, has_blinker=True")


def test_face_score_fields():
    # With a closed mouth (mar=0) the mouth penalty is 1.0, so eye_score == raw average.
    face = _FakeFace(0, right_ear=0.30, left_ear=0.20, mar=0.0)
    result = score_faces([face])
    fs = result.face_scores[0]
    assert math.isclose(fs.right_ear, 0.30, abs_tol=1e-5)
    assert math.isclose(fs.left_ear,  0.20, abs_tol=1e-5)
    # eye_score == 0.25 * penalty; penalty == 1.0 when mouth is closed
    assert fs.eye_score <= 0.25 + 1e-5, f"eye_score should be <= 0.25, got {fs.eye_score}"
    assert fs.face_index == 0
    print("PASS: FaceScore fields (right_ear, left_ear, eye_score, face_index)")


# ---------------------------------------------------------------------------
# Mouth Aspect Ratio tests
# ---------------------------------------------------------------------------

def test_mar_closed_mouth():
    # upper_inner == lower_inner -> vertical = 0 -> MAR = 0
    pts = np.array([[50, 0], [50, 0], [0, 0], [100, 0]], dtype=np.float32)
    mar = mouth_aspect_ratio(pts)
    assert math.isclose(mar, 0.0, abs_tol=1e-5), f"Expected MAR=0.0, got {mar}"
    print("PASS: MAR closed mouth = 0.0")


def test_mar_open_mouth():
    # vertical=60, horizontal=100 -> MAR=0.6
    pts = np.array([[50, 0], [50, 60], [0, 30], [100, 30]], dtype=np.float32)
    mar = mouth_aspect_ratio(pts)
    assert math.isclose(mar, 0.6, abs_tol=1e-5), f"Expected MAR=0.6, got {mar}"
    print("PASS: MAR open mouth = 0.6")


def test_mar_degenerate_horizontal():
    pts = np.zeros((4, 2), dtype=np.float32)   # all at origin
    mar = mouth_aspect_ratio(pts)
    assert mar == 0.0, f"Expected 0.0 for degenerate pts, got {mar}"
    print("PASS: MAR degenerate (all-zeros -> 0.0)")


def test_mouth_penalty_reduces_score_when_enabled():
    """With ENABLE_MOUTH_PENALTY=True, a wide-open mouth must lower the score."""
    ear = 0.28
    face_closed = _FakeFace(0, right_ear=ear, left_ear=ear, mar=0.0)   # mouth closed
    face_open   = _FakeFace(0, right_ear=ear, left_ear=ear, mar=1.0)   # mouth wide open

    score_closed = score_faces([face_closed]).score
    score_open   = score_faces([face_open]).score

    if ENABLE_MOUTH_PENALTY:
        assert score_open < score_closed, (
            f"Open-mouth score ({score_open:.4f}) should be < "
            f"closed-mouth score ({score_closed:.4f}) when ENABLE_MOUTH_PENALTY=True"
        )
        print(f"PASS: mouth penalty active — open {score_open:.3f} < closed {score_closed:.3f}")
    else:
        assert math.isclose(score_open, score_closed, abs_tol=1e-5), (
            "Scores should be equal when ENABLE_MOUTH_PENALTY=False"
        )
        print("PASS: mouth penalty inactive (ENABLE_MOUTH_PENALTY=False) — scores equal")


def test_blinking_flag_ignores_mouth_penalty():
    """blinking is based on raw EAR, not the mouth-penalised score."""
    # EAR above threshold but mouth wide open -> blinking should still be False
    ear  = BLINK_THRESHOLD + 0.05
    face = _FakeFace(0, right_ear=ear, left_ear=ear, mar=2.0)
    result = score_faces([face])
    fs = result.face_scores[0]
    assert not fs.blinking, (
        f"blinking should be False (eyes open) even with wide mouth; ear={ear}"
    )
    print("PASS: blinking=False even with wide-open mouth (EAR is the sole blink signal)")


def test_score_faces_custom_blink_threshold():
    """score_faces respects the blink_threshold parameter (used by GUI sliders)."""
    # EAR 0.22 is above 0.20 (open) but below 0.25 (blinking)
    face = _FakeFace(0, right_ear=0.22, left_ear=0.22, mar=0.0)
    res_low  = score_faces([face], blink_threshold=0.20)
    res_high = score_faces([face], blink_threshold=0.25)
    assert not res_low.face_scores[0].blinking,  "EAR 0.22 >= 0.20 should be open"
    assert     res_high.face_scores[0].blinking, "EAR 0.22 <  0.25 should be blinking"
    print("PASS: score_faces respects custom blink_threshold parameter")


if __name__ == "__main__":
    # EAR
    test_ear_wide_open()
    test_ear_blinking()
    test_ear_fully_closed()
    test_ear_degenerate_horizontal()
    # is_blinking
    test_is_blinking_above_threshold()
    test_is_blinking_below_threshold()
    test_is_blinking_exactly_at_threshold()
    # face_eye_score
    test_face_eye_score_average()
    test_face_eye_score_symmetric()
    # frame_score
    test_frame_score_empty_faces()
    test_frame_score_single_face()
    test_frame_score_takes_minimum()
    test_frame_score_three_faces_one_blinks()
    # score_faces
    test_score_faces_empty()
    test_score_faces_all_open()
    test_score_faces_one_blinker()
    test_face_score_fields()
    # MAR + mouth penalty
    test_mar_closed_mouth()
    test_mar_open_mouth()
    test_mar_degenerate_horizontal()
    test_mouth_penalty_reduces_score_when_enabled()
    test_blinking_flag_ignores_mouth_penalty()
    test_score_faces_custom_blink_threshold()

    print(f"\nAll scoring tests passed.  BLINK_THRESHOLD={BLINK_THRESHOLD}  "
          f"ENABLE_MOUTH_PENALTY={ENABLE_MOUTH_PENALTY}")
