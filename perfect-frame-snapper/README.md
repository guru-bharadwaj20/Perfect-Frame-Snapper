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

---

## Testing

```bash
# All tests are headless — no webcam, no display required (except test_webcam.py)

# Unit tests: scoring logic (23 tests)
python tests/test_scoring.py

# Unit tests: save / buffer logic (6 tests)
python tests/test_main.py

# Smoke tests: detector init + shape contracts (5 tests, loads MediaPipe)
python tests/test_detector.py

# Integration test: opens the webcam and captures 30 frames
python tests/test_webcam.py

# Debug previews (live overlays in an OpenCV window, press Q to quit)
python src/detector.py    # raw landmark overlay
python src/scoring.py     # EAR scores + blink state
```

---

## Project structure

```
perfect-frame-snapper/
├── run.py              ← one-command launcher: python run.py
├── requirements.txt
├── src/
│   ├── config.py       ← all tunables — change here for persistent effect
│   ├── capture.py      ← webcam loop + rolling frame buffer
│   ├── detector.py     ← MediaPipe Face Landmarker wrapper → list[FaceResult]
│   ├── scoring.py      ← EAR + MAR scoring → FrameScore
│   ├── overlay.py      ← OpenCV drawing helpers (shared by main + gui)
│   ├── main.py         ← headless CLI orchestrator
│   └── gui.py          ← Tkinter GUI (live sliders, thumbnail gallery)
├── models/             ← face_landmarker.task (~10 MB, downloaded on first run)
├── output/             ← saved snap_YYYYMMDD_HHMMSS.jpg files
└── tests/
    ├── test_webcam.py
    ├── test_detector.py
    ├── test_scoring.py
    └── test_main.py
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
