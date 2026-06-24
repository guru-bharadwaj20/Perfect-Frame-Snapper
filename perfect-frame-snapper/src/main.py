"""
Perfect-Frame Auto-Snapper — main orchestrator.

Two operating modes:
  MANUAL  Press SPACE to snap.  The tool reaches back through the rolling
          buffer (plus a configurable post-trigger window) and saves the
          highest-scoring frame — the one where all eyes were most open.

  AUTO    The tool watches continuously and saves automatically whenever it
          sees a sustained high-quality moment (all eyes open, no mid-speech
          mouth), then waits for the cooldown before firing again.

Toggle modes with the A key.  All tunables are in src/config.py.

Usage:
    python -m src.main       (from the project root)
    python src/main.py
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

# Ensure project root is on path so src.* imports resolve whether this file
# is run as a module (python -m src.main) or as a script (python src/main.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    BUFFER_SIZE, CAMERA_INDEX, OUTPUT_DIR,
    KEY_SHUTTER, KEY_QUIT, KEY_TOGGLE_AUTO,
    FLASH_SECONDS, SCORE_HISTORY_N,
    POST_TRIGGER_SECONDS,
    AUTO_SNAP_MODE, AUTO_SNAP_THRESHOLD,
    AUTO_SNAP_COOLDOWN, AUTO_SNAP_STABLE_FRAMES,
    MAX_NUM_FACES,
    MIN_FACE_DETECTION_CONFIDENCE,
    MIN_FACE_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

_WINDOW_TITLE = "Perfect-Frame Snapper  |  SPACE=snap  A=auto  Q/ESC=quit"

# ---------------------------------------------------------------------------
# Internal type
# ---------------------------------------------------------------------------

@dataclass
class ScoredFrame:
    """One entry in the scored rolling buffer."""
    score: float
    frame: np.ndarray   # BGR copy — independent of the capture buffer


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------

_GREEN  = ( 30, 200,  30)
_RED    = ( 30,  30, 210)
_ORANGE = ( 30, 160, 220)
_CYAN   = (220, 200,   0)
_WHITE  = (255, 255, 255)
_GREY   = (160, 160, 160)
_BLACK  = (  0,   0,   0)
_DARK   = ( 20,  20,  20)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _write_frame(sf: ScoredFrame, output_dir: Path) -> tuple[bool, str, float]:
    """Write a single ScoredFrame to output_dir as a timestamped JPEG.

    Returns:
        (success, message, score)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = output_dir / filename
    ok = cv2.imwrite(str(save_path), sf.frame)
    if ok:
        return True, f"Saved: {filename}  (score={sf.score:.3f})", sf.score
    return False, f"Write failed -- check permissions on {output_dir}", sf.score


def _save_best(
    buffer: deque[ScoredFrame],
    output_dir: Path,
) -> tuple[bool, str, float]:
    """Scan the buffer for the highest-scoring frame and write it.

    Returns:
        (success, message, best_score)
    """
    if not buffer:
        return False, "Buffer empty -- wait a moment and try again.", 0.0
    best = max(buffer, key=lambda sf: sf.score)
    if best.score == 0.0:
        return False, "No good frame found -- make sure faces are visible.", 0.0
    return _write_frame(best, output_dir)


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def _score_clr(score: float, threshold: float) -> tuple:
    return _GREEN if score >= threshold else _RED


def _draw_score_hud(
    vis: np.ndarray,
    score: float,
    face_count: int,
    buf_filled: int,
    buf_size: int,
    threshold: float,
) -> None:
    """Large centred score + small status line at the top."""
    h, w = vis.shape[:2]
    clr   = _score_clr(score, threshold)
    label = f"SCORE: {score:.3f}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(vis, label, ((w - tw) // 2, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, clr, 2, cv2.LINE_AA)
    status = f"faces: {face_count}   buffer: {buf_filled}/{buf_size}"
    cv2.putText(vis, status, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, _GREY, 1, cv2.LINE_AA)


def _draw_mode_badge(
    vis: np.ndarray,
    auto_mode: bool,
    auto_consecutive: int,
    stable_target: int,
) -> None:
    """Top-right badge showing current mode and auto-snap progress."""
    h, w = vis.shape[:2]
    if auto_mode:
        # Progress bar: filled blocks for consecutive good frames
        filled = min(auto_consecutive, stable_target)
        bar    = "#" * filled + "-" * (stable_target - filled)
        badge  = f"AUTO [{bar}]"
        clr    = _GREEN if auto_consecutive >= stable_target else _CYAN
    else:
        badge = "MANUAL"
        clr   = _GREY
    (tw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(vis, badge, (w - tw - 8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1, cv2.LINE_AA)


def _draw_face_labels(
    vis: np.ndarray,
    faces: list,
    result,
    threshold: float,
) -> None:
    """Per-face score + OPEN/BLINK label near each face's eye region."""
    for fs in result.face_scores:
        face  = faces[fs.face_index]
        r_ctr = face.right_eye_pts.mean(axis=0)
        l_ctr = face.left_eye_pts.mean(axis=0)
        cx    = int((r_ctr[0] + l_ctr[0]) / 2)
        cy    = int((r_ctr[1] + l_ctr[1]) / 2) - 22
        clr   = _score_clr(fs.eye_score, threshold)
        tag   = "BLINK" if fs.blinking else "OPEN"
        line  = f"#{fs.face_index} {fs.eye_score:.2f} [{tag}]"

        # Dark shadow for readability over any background
        cv2.putText(vis, line, (cx - 58, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _BLACK, 3, cv2.LINE_AA)
        cv2.putText(vis, line, (cx - 58, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr,   1, cv2.LINE_AA)

        # Eye landmark dots
        for pt in face.right_eye_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in face.left_eye_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)


def _draw_readiness_bar(
    vis: np.ndarray,
    score_history: deque,
    threshold: float,
) -> None:
    """Thin coloured strip showing per-frame score history above the footer.

    Green = score >= threshold (eyes open).
    Red   = score <  threshold (blink or no face).
    The strip scrolls left as new frames arrive; the right edge is 'now'.
    """
    h, w      = vis.shape[:2]
    bar_h     = 6
    bar_y     = h - 22
    history   = list(score_history)
    n         = len(history)
    if n == 0:
        return
    maxlen    = score_history.maxlen or n
    for i, s in enumerate(history):
        x0 = int(i * w / maxlen)
        x1 = int((i + 1) * w / maxlen)
        cv2.rectangle(vis, (x0, bar_y), (max(x1, x0 + 1), bar_y + bar_h),
                      _score_clr(s, threshold), cv2.FILLED)


def _draw_post_trigger(vis: np.ndarray, remaining: float) -> None:
    """Small countdown shown while the post-trigger window is open."""
    cv2.putText(vis, f"Post-trigger: {remaining:.1f}s remaining",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _ORANGE, 1, cv2.LINE_AA)


def _draw_flash(vis: np.ndarray, msg: str, colour: tuple) -> None:
    """Semi-transparent centred banner for save/error notifications."""
    h, w  = vis.shape[:2]
    bh    = 64
    y1    = h // 2 - bh // 2
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, y1), (w, y1 + bh), _DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    cv2.putText(vis, msg, (max(8, (w - tw) // 2), y1 + (bh + th) // 2),
                font, scale, colour, thick, cv2.LINE_AA)


def _draw_footer(
    vis: np.ndarray,
    auto_mode: bool,
    cooldown_until: float,
    post_trig_secs: float,
    now: float,
) -> None:
    """Control hint strip at the very bottom of the frame."""
    h, w = vis.shape[:2]
    if auto_mode:
        cd   = max(0.0, cooldown_until - now)
        hint = (f"AUTO ON | cooldown {cd:.1f}s | A=manual | Q/ESC=quit"
                if cd > 0 else
                "AUTO ON -- watching for good frame | A=manual | Q/ESC=quit")
    else:
        snap = (f"SPACE=snap (+{post_trig_secs:.0f}s window)"
                if post_trig_secs > 0 else "SPACE=snap")
        hint = f"{snap} | A=auto mode | Q/ESC=quit"
    cv2.putText(vis, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _GREY, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Perfect-Frame Auto-Snapper."""
    from src.capture import FrameBuffer
    from src.detector import FaceDetector
    from src.scoring import BLINK_THRESHOLD, score_faces

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Buffers ──────────────────────────────────────────────────────────────
    scored_buffer: deque[ScoredFrame] = deque(maxlen=BUFFER_SIZE)
    score_history: deque[float]       = deque(maxlen=SCORE_HISTORY_N)

    # ── Runtime state ─────────────────────────────────────────────────────────
    post_trigger_until:  float = 0.0   # monotonic deadline; 0 = inactive
    auto_mode:           bool  = AUTO_SNAP_MODE
    auto_consecutive:    int   = 0     # frames consecutively above AUTO_SNAP_THRESHOLD
    auto_cooldown_until: float = 0.0   # monotonic; don't auto-snap before this

    # ── Flash banner state ────────────────────────────────────────────────────
    flash_msg:    str   = ""
    flash_colour: tuple = _WHITE
    flash_until:  float = 0.0

    # ── Session stats ─────────────────────────────────────────────────────────
    frames_processed: int   = 0
    snaps_saved:      int   = 0
    best_score_saved: float = 0.0

    # ── Helper: fire a save and update flash + session stats ──────────────────
    def _do_save_and_flash(success: bool, msg: str, saved_score: float) -> None:
        nonlocal flash_msg, flash_colour, flash_until, snaps_saved, best_score_saved
        flash_msg    = msg
        flash_colour = _GREEN if success else _ORANGE
        flash_until  = time.monotonic() + FLASH_SECONDS
        if success:
            snaps_saved      += 1
            best_score_saved  = max(best_score_saved, saved_score)

    # ── Startup banner ────────────────────────────────────────────────────────
    print(f"Perfect-Frame Snapper started  [{'AUTO' if auto_mode else 'MANUAL'} mode]")
    print(f"  Output     : {OUTPUT_DIR.resolve()}")
    print(f"  Buffer     : {BUFFER_SIZE} frames  |  post-trigger: {POST_TRIGGER_SECONDS}s")
    print(f"  Blink thr  : {BLINK_THRESHOLD}  |  auto threshold: {AUTO_SNAP_THRESHOLD}")
    print(f"  Auto stable: {AUTO_SNAP_STABLE_FRAMES} frames  |  cooldown: {AUTO_SNAP_COOLDOWN}s")
    print(f"  SPACE=snap  A=toggle auto  Q/ESC=quit")

    buf = FrameBuffer(camera_index=CAMERA_INDEX, maxlen=BUFFER_SIZE)

    with FaceDetector(
        max_num_faces=MAX_NUM_FACES,
        min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as detector:
        buf.open()
        try:
            while True:
                ok, frame = buf.read_frame()
                if not ok:
                    print("Camera read failed -- exiting.")
                    break

                # ── Detect + score ────────────────────────────────────────────
                faces  = detector.detect(frame)
                result = score_faces(faces)

                scored_buffer.append(ScoredFrame(score=result.score, frame=frame.copy()))
                score_history.append(result.score)
                frames_processed += 1

                now = time.monotonic()

                # ── Post-trigger deadline ─────────────────────────────────────
                if 0 < post_trigger_until <= now:
                    post_trigger_until = 0.0
                    ok2, msg, saved = _save_best(scored_buffer, OUTPUT_DIR)
                    _do_save_and_flash(ok2, msg, saved)
                    print(msg)

                # ── Auto-snap ─────────────────────────────────────────────────
                elif auto_mode and post_trigger_until == 0:
                    if result.score >= AUTO_SNAP_THRESHOLD:
                        auto_consecutive += 1
                    else:
                        auto_consecutive = 0

                    if (auto_consecutive >= AUTO_SNAP_STABLE_FRAMES
                            and now >= auto_cooldown_until):
                        # Save the best frame from the stable window
                        recent  = list(scored_buffer)[-AUTO_SNAP_STABLE_FRAMES:]
                        best_sf = max(recent, key=lambda sf: sf.score)
                        ok2, msg, saved = _write_frame(best_sf, OUTPUT_DIR)
                        _do_save_and_flash(ok2, f"Auto: {msg}", saved)
                        print(f"[AUTO] {msg}")
                        auto_consecutive    = 0
                        auto_cooldown_until = now + AUTO_SNAP_COOLDOWN

                # ── Build display ──────────────────────────────────────────────
                vis = frame.copy()
                _draw_face_labels(vis, faces, result, BLINK_THRESHOLD)
                _draw_readiness_bar(vis, score_history, BLINK_THRESHOLD)
                _draw_score_hud(vis, result.score, len(faces),
                                len(scored_buffer), BUFFER_SIZE, BLINK_THRESHOLD)
                _draw_mode_badge(vis, auto_mode, auto_consecutive, AUTO_SNAP_STABLE_FRAMES)
                if post_trigger_until > now:
                    _draw_post_trigger(vis, post_trigger_until - now)
                _draw_footer(vis, auto_mode, auto_cooldown_until, POST_TRIGGER_SECONDS, now)
                if now < flash_until:
                    _draw_flash(vis, flash_msg, flash_colour)

                cv2.imshow(_WINDOW_TITLE, vis)

                # ── Key handling ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == KEY_SHUTTER and not auto_mode and post_trigger_until == 0:
                    if POST_TRIGGER_SECONDS > 0:
                        post_trigger_until = now + POST_TRIGGER_SECONDS
                    else:
                        ok2, msg, saved = _save_best(scored_buffer, OUTPUT_DIR)
                        _do_save_and_flash(ok2, msg, saved)
                        print(msg)

                elif key == KEY_TOGGLE_AUTO:
                    auto_mode        = not auto_mode
                    auto_consecutive = 0
                    print(f"Auto-snap: {'ON' if auto_mode else 'OFF'}")

                elif key in KEY_QUIT:
                    break

        finally:
            buf.release()

    # ── Session summary ────────────────────────────────────────────────────────
    print()
    print("Session summary")
    print(f"  Frames processed : {frames_processed:,}")
    print(f"  Snaps saved      : {snaps_saved}")
    print(f"  Best score saved : {best_score_saved:.3f}")
    print(f"  Output dir       : {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
