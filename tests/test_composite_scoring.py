"""
Unit tests for Feature 2 (composite scoring) and Feature 4 (face weighting).

All tests are headless — no webcam, no display required.

Run from the project root:
    python tests/test_composite_scoring.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scoring import (
    compute_blur_score,
    face_prominence_weight,
    score_faces,
    FrameScore,
)
from src.config import (
    USE_COMPOSITE_SCORING,
    USE_FACE_WEIGHTING,
    COMPOSITE_WEIGHT_EYE,
    COMPOSITE_WEIGHT_BLUR,
    COMPOSITE_WEIGHT_POSE,
)


# ---------------------------------------------------------------------------
# Re-use synthetic face helpers from test_scoring.py
# ---------------------------------------------------------------------------

def _make_eye_pts(ear_target: float, width: float = 100.0) -> np.ndarray:
    v = ear_target * width / 2.0
    w = width
    return np.array([
        [0,     0], [w/3,   v], [2*w/3, v],
        [w,     0], [2*w/3, -v], [w/3,   -v],
    ], dtype=np.float32)


class _FakeFace:
    def __init__(self, face_index, right_ear, left_ear, mar=0.0,
                 eye_x_center=320.0, eye_y_center=240.0,
                 eye_width_scale=1.0):
        self.face_index    = face_index
        self.right_eye_pts = _make_eye_pts(right_ear)
        self.left_eye_pts  = _make_eye_pts(left_ear)
        self.mouth_pts     = np.array([
            [50.0,    0.0],
            [50.0, mar * 100.0],
            [ 0.0, mar * 50.0],
            [100.0, mar * 50.0],
        ], dtype=np.float32)

    def with_eye_positions(self, r_center_x, r_center_y, l_center_x, l_center_y,
                           eye_width=100.0):
        """Return a version of this face with eyes at specific pixel positions."""
        r = self.right_eye_pts.copy()
        l = self.left_eye_pts.copy()
        r_off = np.array([r_center_x - r.mean(axis=0)[0],
                           r_center_y - r.mean(axis=0)[1]], dtype=np.float32)
        l_off = np.array([l_center_x - l.mean(axis=0)[0],
                           l_center_y - l.mean(axis=0)[1]], dtype=np.float32)
        r = r + r_off
        l = l + l_off
        obj = _FakeFace(self.face_index, 0, 0)
        obj.right_eye_pts = r
        obj.left_eye_pts  = l
        obj.mouth_pts     = self.mouth_pts.copy()
        return obj


# ---------------------------------------------------------------------------
# compute_blur_score
# ---------------------------------------------------------------------------

def test_blur_score_sharp_noise_image():
    rng   = np.random.default_rng(42)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    score = compute_blur_score(frame)
    assert score > 0.5, f"Random noise should be sharp, got blur_score={score:.3f}"
    print(f"PASS: compute_blur_score — noise image is sharp ({score:.3f} > 0.5)")


def test_blur_score_solid_colour_image():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    score = compute_blur_score(frame)
    assert score == 0.0, f"Solid colour should be blurry, got blur_score={score:.3f}"
    print("PASS: compute_blur_score — solid colour image = 0.0 (max blur)")


def test_blur_score_range():
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    score = compute_blur_score(frame)
    assert 0.0 <= score <= 1.0
    print(f"PASS: compute_blur_score — result in [0, 1]: {score:.3f}")


# ---------------------------------------------------------------------------
# face_prominence_weight
# ---------------------------------------------------------------------------

def test_prominence_larger_face_wins():
    """Larger face (bigger inter-ocular distance) gets higher weight."""
    # Close-up face: eyes 200 px apart
    close_face = _FakeFace(0, 0.28, 0.28).with_eye_positions(
        r_center_x=220, r_center_y=240,
        l_center_x=420, l_center_y=240,
    )
    # Distant face: eyes 60 px apart
    far_face = _FakeFace(1, 0.28, 0.28).with_eye_positions(
        r_center_x=290, r_center_y=240,
        l_center_x=350, l_center_y=240,
    )
    w_close = face_prominence_weight(close_face, frame_w=640, frame_h=480)
    w_far   = face_prominence_weight(far_face,   frame_w=640, frame_h=480)
    assert w_close > w_far, (
        f"Close face ({w_close:.3f}) should outweigh far face ({w_far:.3f})"
    )
    print(f"PASS: face_prominence_weight — close={w_close:.3f} > far={w_far:.3f}")


def test_prominence_centred_face_wins():
    """Face at frame centre gets higher weight than peripheral face."""
    center_face = _FakeFace(0, 0.28, 0.28).with_eye_positions(
        r_center_x=290, r_center_y=240,
        l_center_x=350, l_center_y=240,
    )
    edge_face = _FakeFace(1, 0.28, 0.28).with_eye_positions(
        r_center_x=10,  r_center_y=10,
        l_center_x=70,  l_center_y=10,
    )
    w_center = face_prominence_weight(center_face, frame_w=640, frame_h=480)
    w_edge   = face_prominence_weight(edge_face,   frame_w=640, frame_h=480)
    assert w_center > w_edge, (
        f"Centred face ({w_center:.3f}) should outweigh edge face ({w_edge:.3f})"
    )
    print(f"PASS: face_prominence_weight — centre={w_center:.3f} > edge={w_edge:.3f}")


def test_prominence_weight_positive():
    face = _FakeFace(0, 0.28, 0.28)
    w = face_prominence_weight(face, frame_w=640, frame_h=480)
    assert w > 0.0
    print(f"PASS: face_prominence_weight — always positive ({w:.6f})")


# ---------------------------------------------------------------------------
# score_faces — new fields always populated
# ---------------------------------------------------------------------------

def test_score_faces_new_fields_present():
    face   = _FakeFace(0, 0.28, 0.28)
    frame  = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = score_faces([face], frame=frame)
    assert hasattr(result, "blur_score"),      "FrameScore missing blur_score"
    assert hasattr(result, "pose_score"),      "FrameScore missing pose_score"
    assert hasattr(result, "composite_score"), "FrameScore missing composite_score"
    print("PASS: FrameScore has blur_score, pose_score, composite_score fields")


def test_score_faces_no_frame_blur_is_zero():
    face   = _FakeFace(0, 0.28, 0.28)
    result = score_faces([face])   # no frame supplied
    assert result.blur_score == 0.0
    print("PASS: blur_score=0.0 when frame not supplied")


def test_score_faces_composite_disabled_score_unchanged():
    """When USE_COMPOSITE_SCORING=False (default), .score == min(eye_score)."""
    face   = _FakeFace(0, 0.28, 0.28)
    frame  = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Explicitly disable composite to confirm default behaviour regardless of config
    result = score_faces([face], frame=frame, use_composite=False)
    # score should equal eye_score (no mouth penalty on closed mouth)
    assert math.isclose(result.score, 0.28, abs_tol=1e-4), (
        f"Expected score≈0.28 (min eye score), got {result.score:.4f}"
    )
    print(f"PASS: composite disabled -> .score = min(eye_score) = {result.score:.4f}")


def test_score_faces_composite_enabled_is_blend():
    """When use_composite=True, .score is a blend and composite_score == .score."""
    face   = _FakeFace(0, 0.28, 0.28)
    frame  = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = score_faces([face], frame=frame, use_composite=True)
    assert math.isclose(result.score, result.composite_score, abs_tol=1e-9), (
        "When use_composite=True, .score must equal .composite_score"
    )
    # Composite includes blur and pose — verify it's computed from the blend
    expected = (
        COMPOSITE_WEIGHT_EYE  * result.face_scores[0].eye_score
        + COMPOSITE_WEIGHT_BLUR * result.blur_score
        + COMPOSITE_WEIGHT_POSE * result.pose_score
    )
    assert math.isclose(result.composite_score, expected, abs_tol=1e-6), (
        f"composite_score mismatch: got {result.composite_score:.4f}, "
        f"expected {expected:.4f}"
    )
    print(f"PASS: composite enabled -> .score = weighted blend ({result.score:.4f})")


# ---------------------------------------------------------------------------
# score_faces — face weighting
# ---------------------------------------------------------------------------

def test_face_weighting_disabled_uses_min():
    """Default (use_face_weighting=False): score = min(eye_score)."""
    faces = [_FakeFace(0, 0.30, 0.30), _FakeFace(1, 0.10, 0.10)]
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    result = score_faces(faces, frame=frame, use_face_weighting=False,
                         use_composite=False)
    assert math.isclose(result.score, 0.10, abs_tol=1e-4), (
        f"Expected min=0.10, got {result.score:.4f}"
    )
    print(f"PASS: face weighting disabled -> score = min = {result.score:.4f}")


def test_face_weighting_enabled_weighted_average():
    """When use_face_weighting=True, a peripheral blinker has less pull."""
    # face 0 (centred, large): good EAR 0.30
    good_face = _FakeFace(0, 0.30, 0.30).with_eye_positions(
        r_center_x=290, r_center_y=240,
        l_center_x=350, l_center_y=240,
    )
    # face 1 (edge, tiny): blinking EAR 0.10
    blink_face = _FakeFace(1, 0.10, 0.10).with_eye_positions(
        r_center_x=5,  r_center_y=5,
        l_center_x=25, l_center_y=5,
    )
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    score_min      = score_faces([good_face, blink_face], frame=frame,
                                 use_face_weighting=False, use_composite=False).score
    score_weighted = score_faces([good_face, blink_face], frame=frame,
                                 use_face_weighting=True,  use_composite=False).score

    assert score_weighted > score_min, (
        f"Weighted score ({score_weighted:.4f}) should be > min score ({score_min:.4f})"
    )
    print(f"PASS: face weighting -> weighted ({score_weighted:.4f}) > min ({score_min:.4f})")


if __name__ == "__main__":
    # Blur
    test_blur_score_sharp_noise_image()
    test_blur_score_solid_colour_image()
    test_blur_score_range()
    # Prominence
    test_prominence_larger_face_wins()
    test_prominence_centred_face_wins()
    test_prominence_weight_positive()
    # FrameScore fields
    test_score_faces_new_fields_present()
    test_score_faces_no_frame_blur_is_zero()
    # Composite
    test_score_faces_composite_disabled_score_unchanged()
    test_score_faces_composite_enabled_is_blend()
    # Face weighting
    test_face_weighting_disabled_uses_min()
    test_face_weighting_enabled_weighted_average()
    print("\nAll composite scoring + face weighting tests passed.")
