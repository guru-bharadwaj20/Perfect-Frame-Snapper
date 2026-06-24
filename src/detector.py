"""
Face detection and facial-landmark extraction using MediaPipe Face Landmarker.

MediaPipe 0.10.x dropped the legacy `solutions` API in favour of a Tasks API
that requires a bundled .task model file.  _ensure_model() downloads it once
to <project-root>/models/ and caches it across runs.

EAR landmark ordering (Soukupová & Čech, 2016) — used identically in scoring.py:

         P2 ─── P3
    P1 ──         ── P4          EAR = (||P2-P6|| + ||P3-P5||)
         P6 ─── P5                     ─────────────────────────
                                             2 · ||P1-P4||

MediaPipe Face Landmarker landmark indices in the above P1-P6 order:

    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
      33  → P1 outer corner (lateral canthus)
     160  → P2 upper-outer lid
     158  → P3 upper-inner lid
     133  → P4 inner corner (medial canthus)
     153  → P5 lower-inner lid
     144  → P6 lower-outer lid

    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
     362  → P1 outer corner
     385  → P2 upper-outer lid
     387  → P3 upper-inner lid
     263  → P4 inner corner
     373  → P5 lower-inner lid
     380  → P6 lower-outer lid

"Left" and "right" follow the subject's anatomy, not the mirror image.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR  = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _ensure_model() -> Path:
    """Download the Face Landmarker model if it isn't already cached."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Face Landmarker model to {_MODEL_PATH} …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Download complete.")
    return _MODEL_PATH


# ---------------------------------------------------------------------------
# EAR landmark index constants
# ---------------------------------------------------------------------------

RIGHT_EYE_IDX: list[int] = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX:  list[int] = [362, 385, 387, 263, 373, 380]

# Mouth points for the Mouth Aspect Ratio (MAR) mid-speech penalty.
# Ordered as: [upper_inner, lower_inner, left_corner, right_corner]
# MAR = ||upper_inner - lower_inner|| / ||left_corner - right_corner||
MOUTH_IDX: list[int] = [13, 14, 61, 291]
#  13  -> upper inner lip centre
#  14  -> lower inner lip centre
#  61  -> left mouth corner
# 291  -> right mouth corner

# Minimum landmark count required before indexing is safe.
_MIN_LANDMARKS = max(max(RIGHT_EYE_IDX), max(LEFT_EYE_IDX), max(MOUTH_IDX)) + 1

# Debug overlay colours (BGR)
_CLR_RIGHT = (0, 255, 255)    # cyan
_CLR_LEFT  = (255, 0, 255)    # magenta
_CLR_TEXT  = (255, 255, 255)  # white


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Detected face with pre-extracted landmark pixel coordinates.

    Attributes:
        face_index:    Position in the list returned by detect() (0-based).
        right_eye_pts: (6, 2) float32 array — pixel (x, y) in RIGHT_EYE_IDX order.
        left_eye_pts:  (6, 2) float32 array — pixel (x, y) in LEFT_EYE_IDX order.
        mouth_pts:     (4, 2) float32 array — pixel (x, y) in MOUTH_IDX order.
    """
    face_index:    int
    right_eye_pts: np.ndarray   # shape (6, 2)
    left_eye_pts:  np.ndarray   # shape (6, 2)
    mouth_pts:     np.ndarray   # shape (4, 2)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """MediaPipe Face Landmarker wrapper (Tasks API, VIDEO running mode).

    VIDEO mode processes frames synchronously with monotonically increasing
    timestamps — the right choice for a continuous webcam loop because it
    enables frame-to-frame tracking that reduces per-frame detection cost.

    Usage:
        with FaceDetector() as detector:
            results = detector.detect(frame)   # list[FaceResult]

    Or without context manager:
        detector = FaceDetector()
        results  = detector.detect(frame)
        detector.close()
    """

    def __init__(
        self,
        max_num_faces: int = 8,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Args:
            max_num_faces:                  Upper bound on simultaneous faces.
            min_face_detection_confidence:  Threshold for initial detection (0–1).
            min_face_presence_confidence:   Threshold for face-presence check (0–1).
            min_tracking_confidence:        Threshold for inter-frame tracking (0–1).
        """
        model_path = _ensure_model()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        # Timestamps must be strictly increasing milliseconds from a fixed origin.
        self._origin_ms = int(time.monotonic() * 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        """Detect faces in a BGR frame and return eye landmark pixel coords.

        Args:
            frame: np.ndarray of shape (H, W, 3) in BGR colour order.

        Returns:
            A (possibly empty) list of FaceResult objects, one per detected
            face.  Returns [] when no faces are found or landmarks are absent.
        """
        h, w = frame.shape[:2]

        # Build a MediaPipe Image from the BGR frame.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonically increasing timestamp in milliseconds.
        timestamp_ms = int(time.monotonic() * 1000) - self._origin_ms

        mp_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not mp_result.face_landmarks:
            return []

        results: list[FaceResult] = []
        for idx, face_lms in enumerate(mp_result.face_landmarks):
            if len(face_lms) < _MIN_LANDMARKS:
                continue  # incomplete landmark set — skip rather than crash
            right_pts  = _extract_pts(face_lms, RIGHT_EYE_IDX, w, h)
            left_pts   = _extract_pts(face_lms, LEFT_EYE_IDX,  w, h)
            mouth_pts  = _extract_pts(face_lms, MOUTH_IDX,     w, h)
            results.append(FaceResult(
                face_index=idx,
                right_eye_pts=right_pts,
                left_eye_pts=left_pts,
                mouth_pts=mouth_pts,
            ))

        return results

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extract_pts(
    landmarks,
    indices: list[int],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Convert normalised MediaPipe landmarks to pixel (x, y) coordinates.

    Args:
        landmarks: Sequence of NormalizedLandmark (with .x/.y in [0, 1]).
        indices:   Indices to extract, returned in that order.
        frame_w:   Frame width in pixels.
        frame_h:   Frame height in pixels.

    Returns:
        float32 array of shape (len(indices), 2).
    """
    return np.array(
        [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in indices],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Debug preview — run directly to visually verify landmark tracking
# ---------------------------------------------------------------------------

def _draw_eye_pts(img: np.ndarray, pts: np.ndarray, colour: tuple) -> None:
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius=3, color=colour, thickness=-1)


def _run_debug_preview() -> None:
    """Live webcam preview with eye-landmark overlay.  Press Q or ESC to quit."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.capture import FrameBuffer

    buf = FrameBuffer(camera_index=0, maxlen=90)

    with FaceDetector() as detector:
        buf.open()
        print("Debug preview running — press Q or ESC in the window to quit.")
        try:
            while True:
                ok, frame = buf.read_frame()
                if not ok:
                    break

                faces = detector.detect(frame)

                # Draw on a copy — keep buffered frames clean.
                vis = frame.copy()
                for face in faces:
                    _draw_eye_pts(vis, face.right_eye_pts, _CLR_RIGHT)
                    _draw_eye_pts(vis, face.left_eye_pts,  _CLR_LEFT)

                label = f"Faces: {len(faces)}  |  cyan=right eye  magenta=left eye"
                cv2.putText(vis, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, _CLR_TEXT, 2, cv2.LINE_AA)
                cv2.putText(vis, "Q / ESC to quit", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _CLR_TEXT, 1, cv2.LINE_AA)

                cv2.imshow("FaceDetector debug", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            buf.release()


if __name__ == "__main__":
    _run_debug_preview()
