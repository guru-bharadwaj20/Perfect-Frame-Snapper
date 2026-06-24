"""
Smoke test for FaceDetector.

Verifies:
  - detector initialises and closes without error
  - detect() returns a list on a blank frame (no faces → empty list)
  - FaceResult arrays have the expected shape when faces are present
    (shape check only — a black frame won't have detectable faces)

Run from the project root:
    python tests/test_detector.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import FaceDetector, FaceResult, RIGHT_EYE_IDX, LEFT_EYE_IDX, MOUTH_IDX


def test_init_and_close():
    d = FaceDetector(max_num_faces=4)
    d.close()
    print("PASS: FaceDetector init/close")


def test_context_manager():
    with FaceDetector() as d:
        pass
    print("PASS: FaceDetector context manager")


def test_blank_frame_returns_empty_list():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with FaceDetector() as d:
        results = d.detect(blank)
    assert isinstance(results, list), "detect() must return a list"
    assert results == [], f"Expected [] for blank frame, got {results}"
    print("PASS: blank frame returns empty list")


def test_index_constants_have_correct_lengths():
    assert len(RIGHT_EYE_IDX) == 6, f"RIGHT_EYE_IDX should have 6 entries, got {len(RIGHT_EYE_IDX)}"
    assert len(LEFT_EYE_IDX)  == 6, f"LEFT_EYE_IDX should have 6 entries, got {len(LEFT_EYE_IDX)}"
    assert len(MOUTH_IDX)     == 4, f"MOUTH_IDX should have 4 entries, got {len(MOUTH_IDX)}"
    print("PASS: eye index constants have 6 entries each; MOUTH_IDX has 4 entries")


def test_face_result_shape_contract():
    """FaceResult arrays must have the correct shapes and dtypes."""
    eye_pts   = np.zeros((6, 2), dtype=np.float32)
    mouth_pts = np.zeros((4, 2), dtype=np.float32)
    fr = FaceResult(
        face_index=0,
        right_eye_pts=eye_pts.copy(),
        left_eye_pts=eye_pts.copy(),
        mouth_pts=mouth_pts.copy(),
    )
    assert fr.right_eye_pts.shape == (6, 2), f"right_eye_pts shape: {fr.right_eye_pts.shape}"
    assert fr.left_eye_pts.shape  == (6, 2), f"left_eye_pts shape: {fr.left_eye_pts.shape}"
    assert fr.mouth_pts.shape     == (4, 2), f"mouth_pts shape: {fr.mouth_pts.shape}"
    assert fr.right_eye_pts.dtype == np.float32
    assert fr.left_eye_pts.dtype  == np.float32
    assert fr.mouth_pts.dtype     == np.float32
    print("PASS: FaceResult shape contract (eyes 6x2, mouth 4x2, all float32)")


if __name__ == "__main__":
    test_init_and_close()
    test_context_manager()
    test_blank_frame_returns_empty_list()
    test_index_constants_have_correct_lengths()
    test_face_result_shape_contract()
    print("\nAll detector smoke tests passed.")
