"""
OpenCV overlay drawing helpers shared by main.py (CLI) and gui.py (GUI).

All functions mutate `vis` in-place.  They never import from config — callers
pass thresholds and state explicitly so live GUI sliders can override defaults.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

# ── Colour palette (BGR) ─────────────────────────────────────────────────────
GREEN  = ( 30, 200,  30)
RED    = ( 30,  30, 210)
ORANGE = ( 30, 160, 220)
CYAN   = (220, 200,   0)
WHITE  = (255, 255, 255)
GREY   = (160, 160, 160)
BLACK  = (  0,   0,   0)
DARK   = ( 20,  20,  20)


def score_clr(score: float, threshold: float) -> tuple:
    return GREEN if score >= threshold else RED


def draw_score_hud(
    vis: np.ndarray,
    score: float,
    face_count: int,
    buf_filled: int,
    buf_size: int,
    threshold: float,
) -> None:
    """Large centred score + small status line at the top."""
    h, w = vis.shape[:2]
    clr   = score_clr(score, threshold)
    label = f"SCORE: {score:.3f}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(vis, label, ((w - tw) // 2, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, clr, 2, cv2.LINE_AA)
    status = f"faces: {face_count}   buffer: {buf_filled}/{buf_size}"
    cv2.putText(vis, status, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)


def draw_mode_badge(
    vis: np.ndarray,
    auto_mode: bool,
    auto_consecutive: int,
    stable_target: int,
) -> None:
    """Top-right badge showing current mode and auto-snap progress."""
    h, w = vis.shape[:2]
    if auto_mode:
        filled = min(auto_consecutive, stable_target)
        bar    = "#" * filled + "-" * (stable_target - filled)
        badge  = f"AUTO [{bar}]"
        clr    = GREEN if auto_consecutive >= stable_target else CYAN
    else:
        badge = "MANUAL"
        clr   = GREY
    (tw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(vis, badge, (w - tw - 8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr, 1, cv2.LINE_AA)


def draw_face_labels(
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
        clr   = score_clr(fs.eye_score, threshold)
        tag   = "BLINK" if fs.blinking else "OPEN"
        line  = f"#{fs.face_index} {fs.eye_score:.2f} [{tag}]"

        cv2.putText(vis, line, (cx - 58, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, BLACK, 3, cv2.LINE_AA)
        cv2.putText(vis, line, (cx - 58, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, clr,   1, cv2.LINE_AA)

        for pt in face.right_eye_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in face.left_eye_pts:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 0, 255), -1)


def draw_readiness_bar(
    vis: np.ndarray,
    score_history: deque,
    threshold: float,
) -> None:
    """Thin coloured strip above the footer showing rolling score history."""
    h, w    = vis.shape[:2]
    bar_h   = 6
    bar_y   = h - 22
    history = list(score_history)
    n       = len(history)
    if n == 0:
        return
    maxlen = score_history.maxlen or n
    for i, s in enumerate(history):
        x0 = int(i * w / maxlen)
        x1 = int((i + 1) * w / maxlen)
        cv2.rectangle(vis, (x0, bar_y), (max(x1, x0 + 1), bar_y + bar_h),
                      score_clr(s, threshold), cv2.FILLED)


def draw_post_trigger(vis: np.ndarray, remaining: float) -> None:
    """Small countdown shown while the post-trigger window is open."""
    cv2.putText(vis, f"Post-trigger: {remaining:.1f}s remaining",
                (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ORANGE, 1, cv2.LINE_AA)


def draw_flash(vis: np.ndarray, msg: str, colour: tuple) -> None:
    """Semi-transparent centred banner for save/error notifications."""
    h, w    = vis.shape[:2]
    bh      = 64
    y1      = h // 2 - bh // 2
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, y1), (w, y1 + bh), DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    cv2.putText(vis, msg, (max(8, (w - tw) // 2), y1 + (bh + th) // 2),
                font, scale, colour, thick, cv2.LINE_AA)


def draw_footer(
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
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREY, 1, cv2.LINE_AA)
