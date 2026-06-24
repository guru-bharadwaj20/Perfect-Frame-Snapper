"""
Face detection and facial-landmark extraction.

STUB — Phase 2 implementation.

This module will wrap MediaPipe Face Mesh to locate all faces in a frame and
return per-face landmark arrays.  The landmarks for the left and right eye
regions are the primary output consumed by scoring.py.

Planned public interface:
    detector = FaceDetector()
    results = detector.process(frame)   # -> list[FaceResult]

Each FaceResult will expose:
    .landmarks   : np.ndarray of shape (468, 3) — normalised (x, y, z)
    .left_eye    : subset of landmarks for the left eye contour
    .right_eye   : subset of landmarks for the right eye contour
"""


class FaceDetector:
    """MediaPipe Face Mesh wrapper — to be implemented in Phase 2."""

    def process(self, frame):
        """Return detected face landmarks for the given BGR frame.

        Args:
            frame: np.ndarray (H, W, 3) in BGR colour order.

        Returns:
            list of FaceResult objects (empty until Phase 2 is implemented).
        """
        raise NotImplementedError("FaceDetector will be implemented in Phase 2.")
