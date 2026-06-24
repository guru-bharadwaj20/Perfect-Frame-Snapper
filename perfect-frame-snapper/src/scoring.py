"""
Frame scoring based on Eye Aspect Ratio (EAR).

EAR formula (Soukupová & Čech, 2016) — landmark ordering defined in detector.py:

         P2 ─── P3
    P1 ──         ── P4          EAR = (||P2-P6|| + ||P3-P5||)
         P6 ─── P5                     ─────────────────────────
                                             2 · ||P1-P4||

Public API (consumed by Phase 4 main.py):

    faces  = detector.detect(frame)         # list[FaceResult]
    result = score_faces(faces)             # FrameScore
    print(result.score, result.has_blinker)

    # Or access raw primitives:
    ear   = eye_aspect_ratio(face.right_eye_pts)
    score = face_eye_score(face)
    ok    = not is_blinking(score)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

# Eyes are considered "open" when average EAR >= this value.
# Typical range: 0.18 (sleepy/squinting) – 0.30 (wide open).
# Blinks dip below 0.15. Starting at 0.20 gives a comfortable margin.
BLINK_THRESHOLD: float = 0.20

# TODO (Phase 2 extension) — mouth-openness penalty
# Requires adding mouth_pts: np.ndarray (shape (4, 2)) to FaceResult in
# detector.py using these MediaPipe indices in this order:
#   [13, 14, 61, 291]
#    13  → upper inner lip centre
#    14  → lower inner lip centre
#    61  → left mouth corner
#   291  → right mouth corner
# MAR = ||pts[0]-pts[1]|| / ||pts[2]-pts[3]||
# Apply a penalty factor of (1 - 0.3 * max(0, MAR - MOUTH_OPEN_THRESHOLD))
# to the face_eye_score when MAR > MOUTH_OPEN_THRESHOLD (~0.5).
# Once detector.py exposes mouth_pts, uncomment the body of _mouth_penalty().
MOUTH_OPEN_THRESHOLD: float = 0.50


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
        eye_score:  Average of right_ear and left_ear.
        blinking:   True when eye_score < BLINK_THRESHOLD.
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
             P1/P4 are the horizontal corners; P2,P3,P5,P6 are lid points.

    Returns:
        EAR as a float.  Higher = more open.  0.0 on degenerate geometry.
    """
    # Vertical distances: upper lid (P2-P6) and (P3-P5)
    vert = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance: corner to corner (P1-P4)
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz < 1e-6:   # degenerate — face nearly sideways or bad detection
        return 0.0
    return float(vert / (2.0 * horiz))


def is_blinking(ear: float) -> bool:
    """Return True when the EAR indicates a closed or closing eye."""
    return ear < BLINK_THRESHOLD


def face_eye_score(face) -> float:
    """Average EAR across both eyes for a single face.

    Args:
        face: FaceResult from detector.py (must have right_eye_pts, left_eye_pts).

    Returns:
        Mean EAR of the two eyes as a float.
    """
    r = eye_aspect_ratio(face.right_eye_pts)
    l = eye_aspect_ratio(face.left_eye_pts)
    return (r + l) * 0.5


def frame_score(faces: list) -> float:
    """Collapse a list of per-face EAR values to a single frame quality score.

    The score is the *minimum* face_eye_score across all detected faces — the
    frame is only as good as its worst-looking person.  Returns 0.0 when the
    face list is empty so that no-face frames never win a buffer scan.

    Args:
        faces: list[FaceResult] from detector.FaceDetector.detect().

    Returns:
        Scalar score ≥ 0.0.  Typically in [0.0, 0.40]; >0.20 means all eyes open.
    """
    if not faces:
        return 0.0
    return min(face_eye_score(f) for f in faces)


# ---------------------------------------------------------------------------
# Rich scoring entry point (used by debug view and Phase 4 main.py)
# ---------------------------------------------------------------------------

def score_faces(faces: list) -> FrameScore:
    """Score every face in a frame and return a structured FrameScore.

    This is the main entry point for Phase 4 and for the debug overlay —
    it bundles the per-face breakdown and the aggregate frame score together.

    Args:
        faces: list[FaceResult] from detector.FaceDetector.detect().

    Returns:
        FrameScore with .score, .has_blinker, and .face_scores populated.
    """
    if not faces:
        return FrameScore()

    face_scores: list[FaceScore] = []
    for face in faces:
        r_ear = eye_aspect_ratio(face.right_eye_pts)
        l_ear = eye_aspect_ratio(face.left_eye_pts)
        avg   = (r_ear + l_ear) * 0.5
        face_scores.append(FaceScore(
            face_index=face.face_index,
            right_ear=r_ear,
            left_ear=l_ear,
            eye_score=avg,
            blinking=is_blinking(avg),
        ))

    best_score  = min(fs.eye_score for fs in face_scores)
    has_blinker = any(fs.blinking for fs in face_scores)

    return FrameScore(
        face_scores=face_scores,
        score=best_score,
        has_blinker=has_blinker,
    )


# ---------------------------------------------------------------------------
# Mouth-penalty stub (activated once detector.py exposes mouth_pts)
# ---------------------------------------------------------------------------

def _mouth_penalty(face) -> float:  # noqa: ARG001
    """Return a [0, 1] multiplier that penalises wide-open mouths.

    Currently a no-op stub.  Activate after Phase 2 is extended to include
    mouth_pts in FaceResult (see TODO comment near MOUTH_OPEN_THRESHOLD above).

    Returns:
        1.0 always until the stub is implemented.
    """
    # TODO: uncomment when mouth_pts is available in FaceResult.
    # mar = (np.linalg.norm(face.mouth_pts[0] - face.mouth_pts[1]) /
    #        max(np.linalg.norm(face.mouth_pts[2] - face.mouth_pts[3]), 1e-6))
    # excess = max(0.0, mar - MOUTH_OPEN_THRESHOLD)
    # return max(0.0, 1.0 - 0.3 * excess)
    return 1.0


# ---------------------------------------------------------------------------
# Debug preview — run directly:  python src/scoring.py
# ---------------------------------------------------------------------------

def _run_debug_preview() -> None:
    """Live webcam preview showing real-time EAR scores and frame quality.

    What to watch for:
      - Frame score stays above BLINK_THRESHOLD (~0.20) with eyes open.
      - Score drops sharply (and turns red) during a blink.
      - With multiple people, one blink tanks the whole frame score.
    Press Q or ESC to quit.
    """
    import sys
    import cv2
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
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
        print("Scoring debug preview — press Q or ESC to quit.")
        try:
            while True:
                ok, frame = buf.read_frame()
                if not ok:
                    break

                faces  = detector.detect(frame)
                result = score_faces(faces)
                vis    = frame.copy()
                h, w   = vis.shape[:2]

                # ── Per-face overlay ────────────────────────────────────────
                for fs in result.face_scores:
                    face = faces[fs.face_index]

                    # Anchor text at the midpoint between the two eye centres.
                    r_centre = face.right_eye_pts.mean(axis=0)
                    l_centre = face.left_eye_pts.mean(axis=0)
                    cx = int((r_centre[0] + l_centre[0]) / 2)
                    cy = int((r_centre[1] + l_centre[1]) / 2) - 20

                    clr = _score_colour(fs.eye_score)
                    tag = "BLINK" if fs.blinking else "OPEN"

                    cv2.putText(vis,
                                f"F{fs.face_index} R:{fs.right_ear:.2f} L:{fs.left_ear:.2f}",
                                (cx - 80, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)
                    cv2.putText(vis,
                                f"eye:{fs.eye_score:.2f}  [{tag}]",
                                (cx - 80, cy + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 1, cv2.LINE_AA)

                    # Draw the 6 eye points for spatial reference.
                    for pt in face.right_eye_pts:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
                    for pt in face.left_eye_pts:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)

                # ── Frame-level score (top-centre, large) ───────────────────
                fscore_txt  = f"FRAME: {result.score:.3f}"
                fscore_clr  = _score_colour(result.score)
                (tw, th), _ = cv2.getTextSize(fscore_txt,
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
                cv2.putText(vis, fscore_txt,
                            ((w - tw) // 2, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, fscore_clr, 2, cv2.LINE_AA)

                # ── HUD footer ──────────────────────────────────────────────
                hud = (f"faces:{len(faces)}  "
                       f"threshold:{BLINK_THRESHOLD:.2f}  "
                       f"{'!! BLINKER !!' if result.has_blinker else 'all open'}")
                cv2.putText(vis, hud, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREY, 1, cv2.LINE_AA)
                cv2.putText(vis, "Q / ESC to quit", (w - 155, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_GREY, 1, cv2.LINE_AA)

                cv2.imshow("Scoring debug", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            buf.release()


if __name__ == "__main__":
    _run_debug_preview()
