"""
Centralized configuration for the Perfect-Frame Auto-Snapper.

Every user-tunable constant lives here.  Edit this file freely — you should
never need to dig into the other source files just to change a threshold,
a keybinding, or a mode toggle.
"""

from pathlib import Path

# ── Capture ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0       # 0 = default webcam; increment if you have several
BUFFER_SIZE  = 90      # rolling window in frames  (~3 s at 30 fps)

# ── Output ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ── Scoring ────────────────────────────────────────────────────────────────────

# EAR threshold: eyes are "open" when average EAR >= this value.
# Typical open-eye range: 0.22 – 0.35.  Blinks dip below 0.15.
# Raise slightly if you get false-positives (e.g. sleepy eyes counted as open).
BLINK_THRESHOLD = 0.20

# Mouth penalty: set False to ignore mouth openness entirely.
# When True, a wide-open mouth (mid-speech) lowers the per-face score so the
# auto-snapper avoids catching someone mid-word.
ENABLE_MOUTH_PENALTY = True

# MAR (Mouth Aspect Ratio) above this triggers the penalty.
# Typical range: 0.3 (slightly open) – 1.0+ (wide open / laughing).
MOUTH_OPEN_THRESHOLD = 0.50

# ── Detector ───────────────────────────────────────────────────────────────────
MAX_NUM_FACES                 = 8    # upper bound on simultaneous faces tracked
MIN_FACE_DETECTION_CONFIDENCE = 0.5  # initial face-detection confidence gate
MIN_FACE_PRESENCE_CONFIDENCE  = 0.5  # per-frame face-presence gate
MIN_TRACKING_CONFIDENCE       = 0.5  # inter-frame tracking confidence gate

# ── Manual shutter (SPACE key) ─────────────────────────────────────────────────
# After SPACE is pressed, keep buffering for this many extra seconds before
# scanning for the best frame.  Catches the "best moment lands just after you
# react" case.  Set to 0.0 for pure pre-trigger behaviour (Phase 4 default).
POST_TRIGGER_SECONDS = 1.0

# ── Auto-snap mode ─────────────────────────────────────────────────────────────
AUTO_SNAP_MODE = False   # start in manual mode; toggle live with the A key

# Score required to count as a "good" frame in auto mode.
# Should be above BLINK_THRESHOLD so genuinely open eyes are required.
AUTO_SNAP_THRESHOLD = 0.26

# Seconds to wait between consecutive auto-snaps (prevents duplicate saves).
AUTO_SNAP_COOLDOWN = 5.0

# How many consecutive above-threshold frames are required before firing.
# Requires a sustained good moment rather than a single lucky frame.
# At 30 fps: 8 frames ≈ 0.27 s.
AUTO_SNAP_STABLE_FRAMES = 8

# ── Display ────────────────────────────────────────────────────────────────────
FLASH_SECONDS   = 2.0   # how long the save / error banner stays on screen
SCORE_HISTORY_N = 90    # number of past scores shown in the readiness bar
                         # (matches BUFFER_SIZE so bar ≈ the full buffer window)

# ── Key bindings ───────────────────────────────────────────────────────────────
KEY_SHUTTER     = ord(" ")         # spacebar — fire manual snap
KEY_QUIT        = {ord("q"), 27}   # lowercase q or ESC
KEY_TOGGLE_AUTO = ord("a")         # toggle auto-snap mode on / off

# ── Calibration (Feature 1) ────────────────────────────────────────────────────
# Path to per-user calibration JSON.  Created by --calibrate (CLI) or the
# "Calibrate" button (GUI).  Falls back to static BLINK_THRESHOLD when absent.
# Delete this file to reset to config defaults.
CALIBRATION_FILE = Path(__file__).parent.parent / "calibration.json"

# ── Composite scoring (Feature 2) ─────────────────────────────────────────────
# When False (default): frame score = min(eye_score) across all faces.
# When True: score = weighted blend of eye quality, blur, and head-pose.
USE_COMPOSITE_SCORING = False
COMPOSITE_WEIGHT_EYE  = 0.5   # weight for EAR-based eye/mouth score
COMPOSITE_WEIGHT_BLUR = 0.3   # weight for Laplacian sharpness score
COMPOSITE_WEIGHT_POSE = 0.2   # weight for head-pose (front-facing) score

# Laplacian variance bounds for blur normalisation.
# variance < BLUR_MIN_VARIANCE  →  blur_score 0.0 (very blurry)
# variance > BLUR_MAX_VARIANCE  →  blur_score 1.0 (very sharp)
BLUR_MIN_VARIANCE = 50.0
BLUR_MAX_VARIANCE = 500.0

# ── Face weighting (Feature 4) ────────────────────────────────────────────────
# When False (default): score = min(eye_score) equally across all faces.
# When True: score = prominence-weighted average, so a small / peripheral
# blinker has reduced influence on the final frame score.
USE_FACE_WEIGHTING = False
FACE_WEIGHT_SIZE   = 0.6   # fraction of prominence from relative face size
FACE_WEIGHT_CENTER = 0.4   # fraction of prominence from proximity to frame centre

# ── Adaptive learning (Feature 3) ─────────────────────────────────────────────
# When False (default): AUTO_SNAP_THRESHOLD stays fixed.
# When True: user Keep / Discard feedback nudges the threshold in memory only
# (never modifies config.py or the calibration file).
ENABLE_ADAPTIVE_LEARNING = False
ADAPTIVE_LEARNING_FILE   = Path(__file__).parent.parent / "feedback_log.json"
ADAPTIVE_LEARNING_WINDOW = 20    # decisions per adjustment cycle
ADAPTIVE_LEARNING_STEP   = 0.01  # threshold delta per completed cycle
