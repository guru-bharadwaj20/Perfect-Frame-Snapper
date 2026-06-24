# Perfect-Frame Auto-Snapper

Continuously buffers the last few seconds of webcam footage and retroactively
saves the single best frame — the one where everyone in shot has their eyes
fully open and nobody is mid-blink or mid-word.

## How it works

A rolling deque keeps ~90 frames (~3 s) in memory.  Every frame is scored
using the Eye Aspect Ratio (EAR) derived from MediaPipe face landmarks.
When a save fires (manually or automatically), the tool scans the buffer and
writes the highest-scoring frame to `output/` as a timestamped JPEG.

**EAR** measures how open each eye is (higher = more open).  A blink dips
below `BLINK_THRESHOLD` (default 0.20).  The frame score is the *minimum*
EAR across all detected faces — one blinker tanks the whole frame.

An optional **mouth penalty** reduces the score slightly when anyone's mouth
is wide open (MAR > `MOUTH_OPEN_THRESHOLD`), helping avoid mid-speech frames.

---

## Quick start

```bash
# 1. Create and activate the virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python -m src.main
```

On first run, the MediaPipe Face Landmarker model (~10 MB) is downloaded
automatically to `models/` and cached for future sessions.

---

## Modes

### Manual mode (default)

Press **SPACE** to fire the shutter.  After the keypress the tool buffers
for an extra `POST_TRIGGER_SECONDS` (default 1 s) to catch a slightly-later
good moment, then saves the best frame from the full window.

Set `POST_TRIGGER_SECONDS = 0` in `config.py` for pure pre-trigger behaviour
(saves from the buffer that existed at the moment you pressed the key).

### Auto-snap mode

Toggle with **A**.  The tool watches for a sustained high-quality moment —
`AUTO_SNAP_STABLE_FRAMES` consecutive frames all above `AUTO_SNAP_THRESHOLD`
(default 8 frames ≈ 0.27 s, threshold 0.26).  When that condition holds, it
saves the best frame of that window, then waits `AUTO_SNAP_COOLDOWN` seconds
(default 5 s) before it can fire again.

This avoids spamming duplicate saves: a single good moment produces one snap,
and the cooldown prevents rapid re-fires.

---

## On-screen overlay

| Element | What it shows |
|---|---|
| **SCORE: 0.287** (top centre) | Live frame score — green >= threshold, red below |
| **MANUAL / AUTO [###---]** (top right) | Current mode; in auto, `#` fill shows stable-frame progress |
| **#0 0.28 [OPEN]** (near each face) | Per-face eye score and blink status |
| Cyan dots | Right-eye EAR landmarks |
| Magenta dots | Left-eye EAR landmarks |
| Coloured strip (near bottom) | Rolling score history — green/red per frame |
| Post-trigger countdown | Shown in orange while the post-trigger window is open |
| Flash banner (centre) | Save confirmation ("Saved: snap_….jpg") or error |
| Footer | Context-sensitive control hints |

---

## Controls

| Key | Action |
|---|---|
| **SPACE** | Fire manual snap (inactive in auto mode) |
| **A** | Toggle auto-snap mode on / off |
| **Q** or **ESC** | Quit and print session summary |

---

## Configuration — `src/config.py`

All tunables are in one file.  The most commonly changed ones:

| Constant | Default | What it does |
|---|---|---|
| `BUFFER_SIZE` | `90` | Rolling window size in frames (~3 s at 30 fps) |
| `CAMERA_INDEX` | `0` | Webcam index (increment for non-default cameras) |
| `OUTPUT_DIR` | `output/` | Where JPEGs are saved |
| `BLINK_THRESHOLD` | `0.20` | EAR below this = blinking.  Raise if you get false-positives |
| `ENABLE_MOUTH_PENALTY` | `True` | Penalise wide-open mouths (mid-speech) |
| `MOUTH_OPEN_THRESHOLD` | `0.50` | MAR above this triggers the penalty |
| `POST_TRIGGER_SECONDS` | `1.0` | Extra buffering after SPACE.  `0` = pure pre-trigger |
| `AUTO_SNAP_MODE` | `False` | Start in auto mode instead of manual |
| `AUTO_SNAP_THRESHOLD` | `0.26` | Score required for an auto-snap candidate |
| `AUTO_SNAP_STABLE_FRAMES` | `8` | Consecutive good frames before auto-firing |
| `AUTO_SNAP_COOLDOWN` | `5.0` | Seconds between auto-snaps |
| `KEY_SHUTTER` | `ord(" ")` | Spacebar |
| `KEY_TOGGLE_AUTO` | `ord("a")` | A key |

---

## Testing

```bash
# Phase 1 — webcam smoke test
python tests/test_webcam.py

# Phase 2 — detector smoke test (headless)
python tests/test_detector.py

# Phase 3 — scoring unit tests (headless)
python tests/test_scoring.py

# Phase 4 — save-logic tests (headless)
python tests/test_main.py

# Phase 2 debug — live landmark overlay
python src/detector.py

# Phase 3 debug — live scoring overlay
python src/scoring.py
```

---

## Project structure

```
perfect-frame-snapper/
├── src/
│   ├── config.py       # All tunables — edit this file to change behaviour
│   ├── capture.py      # Webcam loop + raw rolling buffer
│   ├── detector.py     # MediaPipe Face Landmarker wrapper -> list[FaceResult]
│   ├── scoring.py      # EAR + MAR scoring -> FrameScore
│   └── main.py         # Orchestrator: manual + auto-snap modes, overlay, save
├── models/             # Downloaded MediaPipe model (gitignored)
├── output/             # Saved best-frame JPEGs (gitignored)
├── tests/
│   ├── test_webcam.py
│   ├── test_detector.py
│   ├── test_scoring.py
│   └── test_main.py
└── requirements.txt
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `opencv-python` | 4.13.0.92 | Webcam capture, image I/O, overlay drawing |
| `mediapipe` | 0.10.35 | Face Landmarker model (478 landmarks) |
| `numpy` | 2.5.0 | Array ops for EAR / MAR geometry |

Python 3.9 – 3.13 supported.
