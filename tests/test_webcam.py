"""
Smoke test: open the default webcam, capture 30 frames, verify they are valid.

Run from the project root:
    python tests/test_webcam.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.capture import FrameBuffer


def test_webcam_smoke(n_frames: int = 30) -> None:
    buf = FrameBuffer(camera_index=0, maxlen=90)
    buf.open()

    print(f"Webcam opened. Capturing {n_frames} frames...")

    captured = 0
    try:
        while captured < n_frames:
            ok, frame = buf.read_frame()
            assert ok, f"Frame read failed at frame {captured}"
            assert frame is not None, "Frame is None"
            assert frame.ndim == 3, f"Expected 3D array, got shape {frame.shape}"
            assert frame.shape[2] == 3, f"Expected 3 colour channels, got {frame.shape[2]}"
            assert frame.size > 0, "Frame is empty"
            captured += 1

        sample = buf.buffer[-1]
        h, w, c = sample.shape
        print(f"  Frames captured : {captured}")
        print(f"  Frame dimensions: {w} x {h} px, {c} channels")
        print(f"  Buffer length   : {len(buf.buffer)}")
        print("PASS: webcam smoke test succeeded.")
    finally:
        buf.release()


if __name__ == "__main__":
    test_webcam_smoke()
