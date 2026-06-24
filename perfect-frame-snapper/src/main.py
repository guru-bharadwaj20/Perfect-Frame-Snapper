"""
Perfect-Frame Auto-Snapper — main orchestrator.

Ties capture, detection, and scoring together into a live preview loop.
Press SPACE to snap: the tool reaches back through the rolling buffer and saves
the highest-scoring frame (fewest blinks, most eyes open).

Usage:
    python -m src.main          # from the project root
    python src/main.py          # or directly

Tune the constants directly below — this is the file you'll tweak most.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables — edit these freely
# ---------------------------------------------------------------------------

BUFFER_SIZE   = 90           # rolling window in frames (~3 s at 30 fps)
CAMERA_INDEX  = 0            # 0 = default webcam; change if you have several
OUTPUT_DIR    = Path(__file__).parent.parent / "output"
KEY_SHUTTER   = ord(" ")     # spacebar fires the snap
KEY_QUIT      = {ord("q"), 27}  # lowercase q or ESC
WINDOW_TITLE  = "Perfect-Frame Snapper  |  SPACE = snap   Q/ESC = quit"
FLASH_SECONDS = 2.0          # how long the save/error banner stays on screen

# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

@dataclass
class ScoredFrame:
    """One entry in the scored rolling buffer."""
    score: float
    frame: np.ndarray   # full BGR image, a copy of the capture output


# ---------------------------------------------------------------------------
# Overlay colour palette (BGR)
# ---------------------------------------------------------------------------

_GREEN  = (30,  200,  30)
_RED    = (30,   30, 210)
_ORANGE = (30,  160, 220)
_WHITE  = (255, 255, 255)
_GREY   = (160, 160, 160)
_BLACK  = (  0,   0,   0)
_DARK   = ( 20,  20,  20)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_best(
    buffer: deque[ScoredFrame],
    output_dir: Path,
) -> tuple[bool, str, float]:
    """Scan the scored buffer, pick the best frame, and write it to disk.

    Args:
        buffer:     Scored rolling buffer (may be empty).
        output_dir: Directory to write the JPEG into.

    Returns:
        (success, message, best_score)
    """
    if not buffer:
        return False, "Buffer empty — wait a moment and try again.", 0.0

    best = max(buffer, key=lambda sf: sf.score)

    if best.score == 0.0:
        return False, "No good frame found — make sure faces are visible.", 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = output_dir / filename

    ok = cv2.imwrite(str(save_path), best.frame)
    if ok:
        msg = f"Saved: {filename}   score={best.score:.3f}"
        return True, msg, best.score
    else:
        return False, f"Write failed — check permissions on {output_dir}", best.score


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def _draw_score_hud(
    vis: np.ndarray,
    score: float,
    face_count: int,
    buf_filled: int,
    buf_size:   int,
    blink_threshold: float,
) -> None:
    """Render the live score and status line at the top of the frame."""
    h, w = vis.shape[:2]
    clr = _GREEN if score >= blink_threshold else _RED

    # Large centred score
    label = f"SCORE: {score:.3f}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(vis, label, ((w - tw) // 2, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, clr, 2, cv2.LINE_AA)

    # Smaller status line below score
    status = f"faces: {face_count}   buffer: {buf_filled}/{buf_size}"
    cv2.putText(vis, status, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GREY, 1, cv2.LINE_AA)


def _draw_face_labels(
    vis: np.ndarray,
    faces: list,
    result,
    blink_threshold: float,
) -> None:
    """Draw per-face EAR score and open/blink label near each face's eyes."""
    for fs in result.face_scores:
        face = faces[fs.face_index]

        # Anchor between the two eye centres, slightly above
        r_ctr = face.right_eye_pts.mean(axis=0)
        l_ctr = face.left_eye_pts.mean(axis=0)
        cx = int((r_ctr[0] + l_ctr[0]) / 2)
        cy = int((r_ctr[1] + l_ctr[1]) / 2) - 22

        clr  = _GREEN if not fs.blinking else _RED
        tag  = "BLINK" if fs.blinking else "OPEN"
        line = f"#{fs.face_index}  {fs.eye_score:.2f}  [{tag}]"

        # Thin dark shadow for readability over any background
        cv2.putText(vis, line, (cx - 58, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, _BLACK, 3, cv2.LINE_AA)
        cv2.putText(vis, line, (cx - 58, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, clr, 1, cv2.LINE_AA)

        # Small dots at eye landmarks for spatial reference
        for pt in face.right_eye_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in face.left_eye_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)


def _draw_flash(vis: np.ndarray, msg: str, colour: tuple) -> None:
    """Render a semi-transparent notification banner in the centre of the frame."""
    h, w = vis.shape[:2]
    bh   = 64                         # banner height
    y1   = h // 2 - bh // 2
    y2   = h // 2 + bh // 2

    # Semi-transparent dark background
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, y1), (w, y2), _DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)

    # Centred text
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    tx = max(8, (w - tw) // 2)
    ty = y1 + (bh + th) // 2

    cv2.putText(vis, msg, (tx, ty), font, scale, colour, thick, cv2.LINE_AA)


def _draw_footer(vis: np.ndarray) -> None:
    """Render the control-hint strip at the very bottom of the frame."""
    h, w = vis.shape[:2]
    hint = "SPACE = snap best frame    Q / ESC = quit"
    cv2.putText(vis, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GREY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Perfect-Frame Auto-Snapper."""
    # Ensure output directory exists before the loop starts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Add project root to sys.path so imports work when run as `python src/main.py`
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.capture import FrameBuffer
    from src.detector import FaceDetector
    from src.scoring import BLINK_THRESHOLD, score_faces

    # Scored rolling buffer — separate from FrameBuffer's raw deque
    scored_buffer: deque[ScoredFrame] = deque(maxlen=BUFFER_SIZE)

    # Flash-banner state
    flash_msg:    str   = ""
    flash_colour: tuple = _WHITE
    flash_until:  float = 0.0          # monotonic deadline

    buf = FrameBuffer(camera_index=CAMERA_INDEX, maxlen=BUFFER_SIZE)

    print(f"Starting Perfect-Frame Snapper.")
    print(f"  Output directory : {OUTPUT_DIR.resolve()}")
    print(f"  Buffer size      : {BUFFER_SIZE} frames")
    print(f"  Blink threshold  : {BLINK_THRESHOLD}")
    print(f"  SPACE = snap   Q/ESC = quit")

    with FaceDetector() as detector:
        buf.open()
        try:
            while True:
                ok, frame = buf.read_frame()
                if not ok:
                    print("Camera read failed — exiting.")
                    break

                # ── Score current frame ──────────────────────────────────
                faces  = detector.detect(frame)
                result = score_faces(faces)

                # Store a copy so the deque holds independent images
                scored_buffer.append(
                    ScoredFrame(score=result.score, frame=frame.copy())
                )

                # ── Build display frame ──────────────────────────────────
                vis = frame.copy()

                _draw_score_hud(
                    vis,
                    score=result.score,
                    face_count=len(faces),
                    buf_filled=len(scored_buffer),
                    buf_size=BUFFER_SIZE,
                    blink_threshold=BLINK_THRESHOLD,
                )
                _draw_face_labels(vis, faces, result, BLINK_THRESHOLD)
                _draw_footer(vis)

                # Flash banner while deadline hasn't passed
                if time.monotonic() < flash_until:
                    _draw_flash(vis, flash_msg, flash_colour)

                cv2.imshow(WINDOW_TITLE, vis)

                # ── Key handling ─────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == KEY_SHUTTER:
                    success, msg, _ = _save_best(scored_buffer, OUTPUT_DIR)
                    flash_msg    = msg
                    flash_colour = _GREEN if success else _ORANGE
                    flash_until  = time.monotonic() + FLASH_SECONDS
                    print(msg)

                elif key in KEY_QUIT:
                    break

        finally:
            buf.release()

    print("Exited cleanly.")


if __name__ == "__main__":
    main()
