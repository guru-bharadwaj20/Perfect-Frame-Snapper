"""
Frame scoring based on Eye Aspect Ratio (EAR) and overall frame quality.

STUB — Phase 3 implementation.

This module will consume the landmark output from detector.py and assign each
frame a scalar score.  The frame with the highest score in the rolling buffer
is the one that gets saved.

Scoring factors (planned):
    1. EAR (Eye Aspect Ratio) — primary signal.  Eyes are "open" when EAR
       exceeds a tunable threshold (typically ~0.25).
    2. All-faces-open gate — score is 0 if ANY detected face has a closed eye.
    3. Optional secondary signals: face sharpness (Laplacian variance),
       neutral mouth (low mouth-aspect-ratio), head-pose frontal-ness.

Planned public interface:
    score = score_frame(face_results)   # -> float in [0.0, 1.0]
    ear   = eye_aspect_ratio(eye_landmarks)  # -> float
"""


def eye_aspect_ratio(eye_landmarks) -> float:
    """Compute the Eye Aspect Ratio for a single eye's landmark points.

    Args:
        eye_landmarks: Array-like of 6 (x, y) landmark coordinates following
                       the standard EAR landmark ordering.

    Returns:
        EAR value as a float.  Higher = more open.
    """
    raise NotImplementedError("eye_aspect_ratio will be implemented in Phase 3.")


def score_frame(face_results) -> float:
    """Assign a scalar quality score to a frame given its face detections.

    Args:
        face_results: List of FaceResult objects from detector.FaceDetector.

    Returns:
        Score in [0.0, 1.0].  0.0 means the frame should not be saved.
    """
    raise NotImplementedError("score_frame will be implemented in Phase 3.")
