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
    is_blinking,
    face_eye_score,
    frame_score,
    score_faces,
    BLINK_THRESHOLD,
    FaceScore,
    FrameScore,
)


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
    def __init__(self, face_index: int, right_ear: float, left_ear: float):
        self.face_index    = face_index
        self.right_eye_pts = _make_eye_pts(right_ear)
        self.left_eye_pts  = _make_eye_pts(left_ear)


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
    face = _FakeFace(0, right_ear=0.30, left_ear=0.20)
    result = score_faces([face])
    fs = result.face_scores[0]
    assert math.isclose(fs.right_ear, 0.30, abs_tol=1e-5)
    assert math.isclose(fs.left_ear,  0.20, abs_tol=1e-5)
    assert math.isclose(fs.eye_score, 0.25, abs_tol=1e-5)
    assert fs.face_index == 0
    print("PASS: FaceScore fields (right_ear, left_ear, eye_score, face_index)")


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

    print(f"\nAll scoring tests passed.  BLINK_THRESHOLD = {BLINK_THRESHOLD}")
