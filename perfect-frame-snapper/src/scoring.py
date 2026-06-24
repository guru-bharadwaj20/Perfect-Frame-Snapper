"""
Frame scoring based on Eye Aspect Ratio (EAR) and optional mouth-penalty.

EAR formula (Soukupova & Cech, 2016) — landmark ordering defined in detector.py:

         P2 --- P3
    P1 --         -- P4          EAR = (||P2-P6|| + ||P3-P5||)
         P6 --- P5                     -------------------------
                                             2 * ||P1-P4||

MAR (Mouth Aspect Ratio) — penalises wide-open mid-speech mouths:

    MAR = ||upper_inner - lower_inner|| / ||left_corner - right_corner||
    penalty_factor = max(0.70, 1.0 - 0.30 * max(0, MAR - MOUTH_OPEN_THRESHOLD))

Public API (consumed by main.py):

    faces  = detector.detect(frame)     # list[FaceResult]
    result = score_faces(faces)         # FrameScore
    print(result.score, result.has_blinker)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Ensure project root is on path so 'src.config' resolves whether this module
# is imported as part of the package or run directly as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BLINK_THRESHOLD, ENABLE_MOUTH_PENALTY, MOUTH_OPEN_THRESHOLD


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FaceScore:
    """Per-face scoring breakdown for one frame.

    Attributes:
        face_index: Matches FaceResult.face_index from detector.py.
        right_ear:  EAR for the subject's right eye.
        left_ear:   EAR for the subject's left eye.
        eye_score:  Final per-face quality score (EAR average * mouth penalty).
        blinking:   True when raw EAR average < BLINK_THRESHOLD.
    """
    face_index: int
    right_ear:  float
    left_ear:   float
    eye_score:  float
    blinking:   bool


@dataclass
class FrameScore:
    """Aggregate quality score for a single frame.

    Attributes:
        face_scores: Per-face breakdown, one entry per detected face.
        score:       Minimum eye_score across all faces (0.0 if no faces).
                     One blinker tanks the whole frame.
        has_blinker: True when any face is currently blinking.
    """
    face_scores: list[FaceScore] = field(default_factory=list)
    score:       float = 0.0
    has_blinker: bool  = False


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def eye_aspect_ratio(pts: np.ndarray) -> float:
    """Compute Eye Aspect Ratio for one eye.

    Args:
        pts: (6, 2) float32 array in P1-P6 order as defined in detector.py.

    Returns:
        EAR as a float.  Higher = more open.  0.0 on degenerate geometry.
    """
    vert  = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz < 1e-6:
        return 0.0
    return float(vert / (2.0 * horiz))


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio (MAR).

    Args:
        pts: (4, 2) float32 in [upper_inner, lower_inner, left_corner, right_corner]
             order — matches FaceResult.mouth_pts from detector.py.

    Returns:
        MAR = vertical_opening / horizontal_width.  0.0 = fully closed.
    """
    vertical   = np.linalg.norm(pts[0] - pts[1])
    horizontal = np.linalg.norm(pts[2] - pts[3])
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def is_blinking(ear: float) -> bool:
    """Return True when the EAR indicates a closed or closing eye."""
    return ear < BLINK_THRESHOLD


def face_eye_score(face) -> float:
    """Raw EAR average across both eyes (no mouth penalty).

    Used directly when you need the pure eye signal without the mouth modifier.
    """
    return (eye_aspect_ratio(face.right_eye_pts) + eye_aspect_ratio(face.left_eye_pts)) * 0.5


def frame_score(faces: list) -> float:
    """Minimum per-face eye score across all detected faces.

    Returns 0.0 when the face list is empty so no-face frames never win.
    Does not apply the mouth penalty — use score_faces() for the full pipeline.
    """
    if not faces:
        return 0.0
    return min(face_eye_score(f) for f in faces)


# ---------------------------------------------------------------------------
# Mouth penalty helper
# ---------------------------------------------------------------------------

def _mouth_penalty_factor(mar: float) -> float:
    """Map MAR to a score multiplier in [0.70, 1.0].

    1.0 = mouth closed / slightly open (no penalty).
    0.70 = maximum penalty (wide-open or laughing).
    """
    excess = max(0.0, mar - MOUTH_OPEN_THRESHOLD)
    return max(0.70, 1.0 - 0.30 * excess)


# ---------------------------------------------------------------------------
# Rich scoring entry point
# ---------------------------------------------------------------------------

def score_faces(faces: list) -> FrameScore:
    """Score every face in a frame and return a structured FrameScore.

    Pipeline per face:
      1. Compute right_ear and left_ear via eye_aspect_ratio().
      2. raw_avg = (right_ear + left_ear) / 2
      3. If ENABLE_MOUTH_PENALTY: apply _mouth_penalty_factor(MAR) to raw_avg.
      4. blinking = raw_avg < BLINK_THRESHOLD  (based on eyes only, not mouth).
      5. eye_score = penalised score stored in FaceScore.

    Frame score = min(eye_score) across all faces; 0.0 on empty input.

    Args:
        faces: list[FaceResult] from detector.FaceDetector.detect().

    Returns:
        FrameScore with .score, .has_blinker, and .face_scores populated.
    """
    if not faces:
        return FrameScore()

    face_scores: list[FaceScore] = []
    for face in faces:
        r_ear   = eye_aspect_ratio(face.right_eye_pts)
        l_ear   = eye_aspect_ratio(face.left_eye_pts)
        raw_avg = (r_ear + l_ear) * 0.5

        if ENABLE_MOUTH_PENALTY:
            mar     = mouth_aspect_ratio(face.mouth_pts)
            penalty = _mouth_penalty_factor(mar)
        else:
            penalty = 1.0

        eye_score = raw_avg * penalty

        face_scores.append(FaceScore(
            face_index=face.face_index,
            right_ear=r_ear,
            left_ear=l_ear,
            eye_score=eye_score,
            blinking=is_blinking(raw_avg),   # based on eyes only, not mouth
        ))

    best_score  = min(fs.eye_score for fs in face_scores)
    has_blinker = any(fs.blinking for fs in face_scores)

    return FrameScore(
        face_scores=face_scores,
        score=best_score,
        has_blinker=has_blinker,
    )


# ---------------------------------------------------------------------------
# Debug preview — run directly:  python src/scoring.py
# ---------------------------------------------------------------------------

def _run_debug_preview() -> None:
    """Live webcam preview showing real-time EAR scores and frame quality.

    What to watch for:
      - Frame score stays green (above BLINK_THRESHOLD) with eyes open.
      - Score drops and turns red during a blink.
      - Score also drops slightly when mouth is wide open (if ENABLE_MOUTH_PENALTY).
      - With multiple people, one blink tanks the whole frame score.
    Press Q or ESC to quit.
    """
    import cv2
    from src.capture import FrameBuffer
    from src.detector import FaceDetector

    CLR_GREEN = (0, 220, 0)
    CLR_RED   = (0, 0, 220)
    CLR_WHITE = (255, 255, 255)
    CLR_GREY  = (160, 160, 160)

    def _score_colour(score: float) -> tuple:
        return CLR_GREEN if score >= BLINK_THRESHOLD else CLR_RED

    buf = FrameBuffer(camera_index=0, maxlen=90)

    with FaceDetector() as detector:
        buf.open()
        print("Scoring debug preview -- press Q or ESC to quit.")
        try:
            while True:
                ok, frame = buf.read_frame()
                if not ok:
                    break

                faces  = detector.detect(frame)
                result = score_faces(faces)
                vis    = frame.copy()
                h, w   = vis.shape[:2]

                for fs in result.face_scores:
                    face    = faces[fs.face_index]
                    r_ctr   = face.right_eye_pts.mean(axis=0)
                    l_ctr   = face.left_eye_pts.mean(axis=0)
                    cx      = int((r_ctr[0] + l_ctr[0]) / 2)
                    cy      = int((r_ctr[1] + l_ctr[1]) / 2) - 20
                    clr     = _score_colour(fs.eye_score)
                    tag     = "BLINK" if fs.blinking else "OPEN"

                    cv2.putText(vis,
                                f"F{fs.face_index} R:{fs.right_ear:.2f} L:{fs.left_ear:.2f}",
                                (cx - 80, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)
                    cv2.putText(vis,
                                f"score:{fs.eye_score:.2f} [{tag}]",
                                (cx - 80, cy + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 1, cv2.LINE_AA)

                    for pt in face.right_eye_pts:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
                    for pt in face.left_eye_pts:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

                fscore_txt  = f"FRAME: {result.score:.3f}"
                fscore_clr  = _score_colour(result.score)
                (tw, th), _ = cv2.getTextSize(fscore_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
                cv2.putText(vis, fscore_txt, ((w - tw) // 2, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, fscore_clr, 2, cv2.LINE_AA)

                hud = (f"faces:{len(faces)}  threshold:{BLINK_THRESHOLD:.2f}  "
                       f"mouth_penalty:{'ON' if ENABLE_MOUTH_PENALTY else 'OFF'}  "
                       f"{'!! BLINKER !!' if result.has_blinker else 'all open'}")
                cv2.putText(vis, hud, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_GREY, 1, cv2.LINE_AA)

                cv2.imshow("Scoring debug", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            buf.release()


if __name__ == "__main__":
    _run_debug_preview()
