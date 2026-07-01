# Perfect-Frame Auto-Snapper

Continuously buffers the last few seconds of webcam footage and retroactively
saves the **single best frame** — the one where everyone in shot has their eyes
fully open and nobody is mid-blink or mid-word.

No manual timing. No burst-mode sifting. One great photo, automatically.

---

## Quick start

```bash
# 1. Create and activate the virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the GUI  ← recommended for everyday use
python run.py

# 4. Or run the headless CLI
python -m src.main
```

On first run, the MediaPipe Face Landmarker model (~10 MB) downloads
automatically to `models/` and is cached for every future session.

---

## How it works

### Scoring pipeline

```
Webcam frame (BGR)
        │
        ▼
MediaPipe Face Landmarker  ──  478 landmarks per detected face
        │
        ▼  for each detected face:
┌───────────────────────────────────────────────────────────────┐
│  6 right-eye landmarks  →  EAR_right ──┐                      │
│  6 left-eye landmarks   →  EAR_left   ─┴──→  raw_avg          │
│                                                   │            │
│  4 mouth landmarks  →  MAR  →  penalty ───────────×           │
│                                (if mouth penalty on)  │        │
│                                                       ▼        │
│                                               eye_score        │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Frame score = min(eye_score) across all detected faces
              ↑
    One blinker tanks the whole frame — 0.0 if no faces visible
```

**EAR** (Eye Aspect Ratio) measures how open each eye is.
A blink dips below `BLINK_THRESHOLD` (default 0.20).

**MAR** (Mouth Aspect Ratio) optionally penalises wide-open mouths (mid-speech)
by up to 30%, capped so the minimum per-face contribution is 0.70 × raw_avg.

### Module structure

```
src/config.py   ← all tunables; edit this file to change behaviour
    ▲  ▲  ▲  ▲
    │  │  │  └── src/capture.py   webcam loop, rolling frame buffer
    │  │  └───── src/detector.py  MediaPipe wrapper → list[FaceResult]
    │  └──────── src/scoring.py   EAR + MAR scoring → FrameScore
    │                 ▲
    └── src/overlay.py            OpenCV drawing helpers (shared)
              ▲            ▲
        src/main.py   src/gui.py
         (CLI mode)   (GUI mode)
              │            │
              └─────┬──────┘
                    ▼
            output/snap_*.jpg
```

---

## Modes

### Manual mode (default)

```
Webcam ──→ Detect ──→ Score ──→ Rolling buffer (last 90 frames, ~3 s)
                                           │
                               Press SPACE / Snap button
                                           │
                      POST_TRIGGER_SECONDS > 0? ──yes──→ Keep buffering N s
                                           │                      │
                                          no               Deadline passes
                                           │                      │
                                           └──────────────────────┘
                                                         │
                                             Scan buffer → best frame
                                                         │
                                                         ▼
                                               output/snap_*.jpg
```

Press **SPACE** (or the Snap button in the GUI) to fire the shutter.
The tool reaches back through the buffer and saves the highest-scoring frame.

With `POST_TRIGGER_SECONDS > 0` (default 1 s) the buffer keeps filling for
that extra window after the keypress, so you can snap *before* everyone is
fully composed and still catch the great moment that follows.

Set `POST_TRIGGER_SECONDS = 0` for pure pre-trigger (saves whatever is in the
buffer at the instant you press the key).

### Auto-snap mode  (toggle with **A** or the mode button)

```
Webcam ──→ Detect ──→ Score
                         │
              score ≥ AUTO_SNAP_THRESHOLD? ───no──→ reset consecutive counter
                         │
                        yes
                         │
              consecutive_good_frames++
                         │
     consecutive_good_frames ≥ AUTO_SNAP_STABLE_FRAMES (default 8)?
                         │
                        yes  AND  cooldown timer expired?
                         │
                         ▼
         Save best of the last 8 frames  →  output/snap_*.jpg
         Reset counter; start cooldown (AUTO_SNAP_COOLDOWN s)
```

The tool watches continuously and saves automatically when it sees a sustained
high-quality moment — `AUTO_SNAP_STABLE_FRAMES` consecutive frames (default 8,
≈ 0.25 s at 30 fps) all above `AUTO_SNAP_THRESHOLD` (default 0.26).

The **stable-frame gate** prevents transient spikes from firing a save.
The **cooldown** (default 5 s) prevents duplicate saves of the same moment.

---

## GUI guide

```
┌─ Perfect-Frame Auto-Snapper ─────────────────────────────────────────────┐
│                                          │                                │
│   ┌──────────────────────────────────┐  │  SCORE: 0.287  (green/red)     │
│   │                                  │  │  faces: 1   buf: 87/90          │
│   │   Live webcam feed (640×480)     │  ├──────────────────────────────  │
│   │   with all Phase 5 overlays      │  │  [Snap [SPACE]]  [AUTO: OFF]    │
│   │                                  │  ├──────────────────────────────  │
│   │   • SCORE top-centre             │  │  Blink threshold     0.20  ──  │
│   │   • MANUAL / AUTO badge          │  │  Auto-snap threshold 0.26  ──  │
│   │   • Per-face #N score [OPEN]     │  │  Auto-snap cooldown  5.0s  ──  │
│   │   • Cyan / magenta eye dots      │  │  Post-trigger window 1.0s  ──  │
│   │   • Green/red history strip      │  │  ☑ Mouth penalty               │
│   │   • Post-trigger countdown       │  ├──────────────────────────────  │
│   │   • Flash banner on save         │  │  Last saved:                   │
│   │                                  │  │  ┌──────────┐  snap_…jpg       │
│   └──────────────────────────────────┘  │  │ thumbnail│  Score: 0.312    │
│                                          │  └──────────┘                  │
│                                          │  [Open output folder]          │
│                                          ├──────────────────────────────  │
│                                          │  Frames: 1,234  Snaps: 3       │
│                                          │  [Quit  [Q / ESC]]             │
└──────────────────────────────────────────────────────────────────────────┘
```

### GUI controls reference

| Control | What it does |
|---|---|
| **Snap [SPACE]** | Fire the manual shutter (greyed out in auto mode) |
| **AUTO: OFF / ON** | Toggle auto-snap mode |
| **Blink threshold** slider | EAR below this = blinking; also controls overlay colours |
| **Auto-snap threshold** slider | Minimum score for an auto-snap candidate frame |
| **Auto-snap cooldown** slider | Seconds of silence after each auto-snap |
| **Post-trigger window** slider | Extra seconds to buffer after SPACE; 0 = pure pre-trigger |
| **Mouth penalty** checkbox | Penalise frames where someone's mouth is wide open |
| **Open output folder** | Opens `output/` in Explorer / Finder |
| **Quit [Q / ESC]** | Releases camera, prints session summary, closes window |

Slider changes take effect **immediately** — no restart needed.

---

## CLI mode

The original headless interface is still fully functional:

```bash
python -m src.main
```

All keyboard shortcuts work the same way in both modes.

### Controls (CLI and GUI)

| Key | Action |
|---|---|
| **SPACE** | Fire manual snap |
| **A** | Toggle auto-snap mode on / off |
| **Q** or **ESC** | Quit and print session summary |

---

## On-screen overlay

| Element | What it shows |
|---|---|
| **SCORE: 0.287** (top centre) | Live frame score — green ≥ threshold, red below |
| **MANUAL / AUTO [###---]** (top right) | Current mode; `#` fill shows stable-frame progress |
| **#0 0.28 [OPEN]** (near each face) | Per-face eye score and blink status |
| Cyan dots | Right-eye EAR landmarks (6 per face) |
| Magenta dots | Left-eye EAR landmarks (6 per face) |
| Coloured strip (near bottom) | Rolling score history — green/red per frame |
| Orange countdown text | Visible while the post-trigger window is open |
| Flash banner (centre) | Save confirmation ("Saved: snap_….jpg  score=0.312") or error |
| Footer | Context-sensitive control hints |

---

## Configuration — `src/config.py`

All tunables live in one file. Change them for a persistent effect; use the
GUI sliders for live, per-session adjustments.

| Constant | Default | What it does |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index — increment for non-default cameras |
| `BUFFER_SIZE` | `90` | Rolling window size in frames (~3 s at 30 fps) |
| `OUTPUT_DIR` | `output/` | Where timestamped JPEGs are saved |
| `BLINK_THRESHOLD` | `0.20` | EAR below this = blinking. Raise if you get false-positives |
| `ENABLE_MOUTH_PENALTY` | `True` | Penalise wide-open mouths (mid-speech) |
| `MOUTH_OPEN_THRESHOLD` | `0.50` | MAR above this triggers the penalty |
| `MAX_NUM_FACES` | `8` | Maximum faces tracked simultaneously |
| `POST_TRIGGER_SECONDS` | `1.0` | Extra buffer time after SPACE. `0` = pure pre-trigger |
| `AUTO_SNAP_MODE` | `False` | Start in auto mode instead of manual |
| `AUTO_SNAP_THRESHOLD` | `0.26` | Score required for an auto-snap candidate |
| `AUTO_SNAP_STABLE_FRAMES` | `8` | Consecutive good frames before auto-firing |
| `AUTO_SNAP_COOLDOWN` | `5.0` | Seconds between auto-snaps |
| `FLASH_SECONDS` | `2.0` | How long the save-confirmation banner stays on screen |
| `SCORE_HISTORY_N` | `90` | Width of the rolling history strip |

### Advanced / opt-in features (all default to OFF)

Features 2–4 below are controlled by boolean flags in `src/config.py` that
default to `False`, so the app behaves exactly as before when you haven't
touched them.  Feature 1 (calibration) is activated by running `--calibrate`
once; no config flag is required.

#### Feature 1 — Per-user adaptive threshold calibration

This feature is **activated by running `--calibrate` once** (CLI) or clicking
the Calibrate button (GUI), not by a config flag.  The result is stored
automatically; no `src/config.py` change is needed.

| Item | Value | Notes |
|---|---|---|
| `CALIBRATION_FILE` | `calibration.json` (project root) | Written by `--calibrate`; loaded on every subsequent launch |
| To disable | — | Delete `calibration.json` to revert to the static `BLINK_THRESHOLD` |

**How to use (CLI):**

```bash
python -m src.main --calibrate
```

A 4-second live window guides you to keep your eyes naturally open.
The tool measures your personal resting EAR, derives a threshold 20 % below
that baseline, and writes `calibration.json` to the project root.  On every
future launch the saved threshold is loaded automatically.  Delete the file
to revert to the static `BLINK_THRESHOLD` default.

**How to use (GUI):** Click the **"Calibrate (set personal EAR baseline)"**
button in the right panel.  The live video briefly shows a calibration
overlay, then updates the blink-threshold slider to your personal value.

#### Feature 2 — Composite frame quality scoring

| Constant | Default | What it does |
|---|---|---|
| `USE_COMPOSITE_SCORING` | `False` | When `True`, replaces min(EAR) with a weighted blend |
| `COMPOSITE_WEIGHT_EYE` | `0.5` | Weight for eye/mouth quality |
| `COMPOSITE_WEIGHT_BLUR` | `0.3` | Weight for Laplacian sharpness score |
| `COMPOSITE_WEIGHT_POSE` | `0.2` | Weight for head-pose front-facing score |
| `BLUR_MIN_VARIANCE` | `50.0` | Laplacian variance mapped to blur_score = 0.0 |
| `BLUR_MAX_VARIANCE` | `500.0` | Laplacian variance mapped to blur_score = 1.0 |

When enabled, `FrameScore.score` becomes
`0.5 * eye_score + 0.3 * blur_score + 0.2 * pose_score`.
`FrameScore` now always exposes `.blur_score`, `.pose_score`, and
`.composite_score` for inspection regardless of this flag.

#### Feature 3 — Adaptive learning from keep/discard feedback

| Constant | Default | What it does |
|---|---|---|
| `ENABLE_ADAPTIVE_LEARNING` | `False` | Show Keep / Discard buttons in GUI after each snap |
| `ADAPTIVE_LEARNING_FILE` | `feedback_log.json` | Append-only decision log |
| `ADAPTIVE_LEARNING_WINDOW` | `20` | Decisions per adjustment cycle |
| `ADAPTIVE_LEARNING_STEP` | `0.01` | Threshold delta per cycle |

After each snap the GUI shows **"Keep"** and **"Discard"** buttons.
Every `ADAPTIVE_LEARNING_WINDOW` decisions the auto-snap threshold slider is
nudged in memory (never written back to `config.py`):

- Acceptance rate < 50 % → raise threshold (be pickier)
- Acceptance rate > 80 % → lower threshold (be more relaxed)
- Otherwise → no change

The adjustment accumulates across sessions via `feedback_log.json`.

#### Feature 4 — Face weighting for multi-face frames

| Constant | Default | What it does |
|---|---|---|
| `USE_FACE_WEIGHTING` | `False` | Prominence-weighted average instead of raw min() |
| `FACE_WEIGHT_SIZE` | `0.6` | Fraction of prominence from relative face size |
| `FACE_WEIGHT_CENTER` | `0.4` | Fraction from proximity to frame centre |

When enabled, a small or peripheral blinker has proportionally less influence
on the frame score than the large, centred primary subject.

---

## Testing

```bash
# All tests are headless — no webcam, no display required (except test_webcam.py)

# Core unit tests (unchanged — verify no regressions)
python tests/test_scoring.py      # 23 tests: EAR/MAR logic
python tests/test_main.py         # 6 tests: save / buffer logic
python tests/test_detector.py     # 5 tests: MediaPipe shape contracts

# New feature tests
python tests/test_calibration.py       # 8 tests: calibration round-trip + logic
python tests/test_composite_scoring.py # 12 tests: blur/pose/weighting
python tests/test_feedback.py          # 12 tests: adaptive learning heuristic

# Integration test: opens the webcam and captures 30 frames
python tests/test_webcam.py

# Debug previews (live overlays in an OpenCV window, press Q to quit)
python src/detector.py    # raw landmark overlay
python src/scoring.py     # EAR scores + blink state
```

---

## Evaluation / benchmarking (Feature 5)

`tests/evaluate.py` measures how well the algorithm's auto-selected frame
matches a human-labelled ground truth.

### Dataset layout

```
tests/eval_data/
    labels.json          <- ground-truth file
    clip_001/            <- one folder per recorded clip
        frame_0001.jpg
        frame_0002.jpg
        ...
    clip_002/
        ...
```

### labels.json format

```json
[
  {
    "clip":       "clip_001",
    "best_frame": 47,
    "note":       "optional description"
  }
]
```

`best_frame` is **1-based** (first image in the sorted folder = frame 1).

### Adding a clip

```bash
# Extract frames from a video (e.g. at 30 fps)
ffmpeg -i my_video.mp4 tests/eval_data/clip_002/frame_%04d.jpg

# Watch the clip, pick the best frame, add to labels.json
# { "clip": "clip_002", "best_frame": 53 }
```

### Running the evaluation

```bash
python tests/evaluate.py                          # default data directory
python tests/evaluate.py --data tests/eval_data   # explicit path
python tests/evaluate.py --tol 15                 # stricter near-match tolerance
python tests/evaluate.py --composite              # evaluate with composite scoring on
python tests/evaluate.py --face-weighting         # evaluate with face weighting on
```

**Output:**

```
Evaluating 2 clip(s) from tests/eval_data
  Tolerance  : +/-30 frames for near-match
  Composite  : OFF
  Weighting  : OFF

  clip_001               gt=  47  algo=  47  score=0.312  fps=  68.1  [EXACT]
  clip_002               gt=  53  algo=  58  score=0.298  fps=  65.7  [NEAR  (+/-5)]

==============================================================
Results over 2 clip(s):
  Exact match rate   : 1/2 = 50.0%
  Near match rate    : 2/2 = 100.0%  (tol=30 frames)
  Mean frame distance: 2.5 frames
  Mean score chosen  : 0.305
  Mean scoring FPS   : 66.9 fps
==============================================================
```

---

## Project structure

```
perfect-frame-snapper/
├── run.py                  <- one-command launcher: python run.py
├── requirements.txt
├── calibration.json        <- created by --calibrate; delete to reset (gitignored)
├── feedback_log.json       <- adaptive learning log; delete to reset (gitignored)
├── src/
│   ├── config.py           <- all tunables — change here for persistent effect
│   ├── capture.py          <- webcam loop + rolling frame buffer
│   ├── detector.py         <- MediaPipe Face Landmarker wrapper -> list[FaceResult]
│   ├── scoring.py          <- EAR + MAR + blur + pose scoring -> FrameScore
│   ├── overlay.py          <- OpenCV drawing helpers (shared by main + gui)
│   ├── main.py             <- headless CLI orchestrator  (--calibrate flag)
│   ├── gui.py              <- Tkinter GUI (Calibrate button, Keep/Discard buttons)
│   ├── calibration.py      <- Feature 1: per-user EAR calibration
│   └── feedback.py         <- Feature 3: adaptive threshold from keep/discard log
├── models/                 <- face_landmarker.task (~10 MB, downloaded on first run)
├── output/                 <- saved snap_YYYYMMDD_HHMMSS.jpg files
└── tests/
    ├── test_webcam.py
    ├── test_detector.py
    ├── test_scoring.py
    ├── test_main.py
    ├── test_calibration.py      <- Feature 1 tests
    ├── test_composite_scoring.py <- Features 2 & 4 tests
    ├── test_feedback.py         <- Feature 3 tests
    ├── evaluate.py              <- Feature 5: benchmarking script
    └── eval_data/
        ├── labels.json          <- ground-truth labels (add your clips here)
        └── clip_001/            <- sample placeholder clip
```

---

## Optional packaging — PyInstaller

To distribute the app as a standalone `.exe` (no Python or venv required):

```bash
pip install pyinstaller

pyinstaller --onefile --windowed \
  --add-data "models/face_landmarker.task;models" \
  --hidden-import mediapipe \
  --hidden-import PIL._tkinter_finder \
  run.py
```

The result appears in `dist/run.exe`.

**Caveats:**
- The bundled binary is typically 200–400 MB due to MediaPipe and OpenCV.
- `--windowed` suppresses the terminal on Windows; remove it to keep the
  session-summary output visible in a console window.
- The `--add-data` flag bundles the model file so the download step is skipped
  at runtime. Use `;` as separator on Windows, `:` on macOS/Linux.
- Test the packaged binary on a machine without Python installed before sharing.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `opencv-python` | 4.13.0.92 | Webcam capture, image I/O, overlay drawing |
| `mediapipe` | 0.10.35 | Face Landmarker model (478 landmarks, Tasks API) |
| `numpy` | 2.5.0 | Array ops for EAR / MAR geometry |
| `Pillow` | ≥ 9.0 | OpenCV BGR frame → Tkinter PhotoImage conversion (GUI only) |

Python 3.9–3.13 supported.  Pillow is a transitive dependency of MediaPipe and
is present in the venv automatically; it is listed explicitly for clarity.
