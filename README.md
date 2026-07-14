# Perfect-Frame Auto-Snapper

![Demo](Demo.png)

[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen)](tests)

Continuously buffers the last few seconds of webcam footage and retroactively
saves the **single best frame** — the one where everyone in shot has their
eyes fully open and nobody is mid-blink or mid-word. No manual timing, no
sifting through a burst of near-identical shots. One great photo, automatically.

---

## Features

- **Real-time face scoring** via MediaPipe's 478-point Face Landmarker
- **Blink detection** using Eye Aspect Ratio (EAR), with an optional Mouth
  Aspect Ratio (MAR) penalty for mid-speech frames
- **Manual mode** — press SPACE and the tool reaches back through its buffer
  to save the best recent frame (with an optional post-trigger window)
- **Auto-snap mode** — watches continuously and saves the instant it sees a
  sustained, high-quality moment
- **Per-user calibration** for a personalized blink threshold
- **Composite quality scoring** (blur + head pose + eye/mouth) and
  **face-weighting** for multi-person shots — both opt-in
- **Adaptive learning** from Keep/Discard feedback to self-tune the auto-snap
  threshold over time
- A lightweight **Tkinter GUI** with live sliders, plus a headless CLI

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the GUI (recommended)
python run.py

# 4. Or run the headless CLI
python -m src.main
```

On first run, the MediaPipe Face Landmarker model (~10 MB) downloads
automatically to `models/` and is cached for every future session.

---

## How it works

Each webcam frame is passed through MediaPipe to detect faces and their
landmarks. For every detected face, eye-openness (EAR) and, optionally, mouth
openness (MAR) are combined into a per-face score; the frame's overall score
is the **minimum across all faces** — one blinker tanks the whole frame. Frames
are held in a rolling buffer (default ~3 s), and when you trigger a snap
(manually or automatically), the highest-scoring frame in that window is
saved to `output/`.

| Mode | Behavior |
|---|---|
| **Manual** (default) | Press **SPACE** to fire the shutter. The buffer keeps filling briefly afterward (`POST_TRIGGER_SECONDS`), then the best frame in the whole window is saved. |
| **Auto-snap** (toggle with **A**) | Saves automatically once a sustained run of high-quality frames (`AUTO_SNAP_STABLE_FRAMES`) clears `AUTO_SNAP_THRESHOLD`, subject to a cooldown to avoid duplicate saves. |

### Module structure

```
src/config.py     all tunables — change here for persistent effect
src/capture.py    webcam loop + rolling frame buffer
src/detector.py   MediaPipe wrapper -> list[FaceResult]
src/scoring.py    EAR / MAR / blur / pose scoring -> FrameScore
src/overlay.py    shared OpenCV drawing helpers
src/main.py       headless CLI orchestrator
src/gui.py        Tkinter GUI
```

---

## Controls

| Key | Action |
|---|---|
| **SPACE** | Fire a manual snap |
| **A** | Toggle auto-snap mode |
| **Q** / **ESC** | Quit and print a session summary |

The GUI mirrors these controls with buttons and adds live sliders for every
threshold, a Calibrate button, and a thumbnail preview of the last saved shot.

---

## Configuration

All tunables live in `src/config.py`; GUI sliders adjust the same values live,
per session. The most commonly changed settings:

| Constant | Default | What it does |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `BUFFER_SIZE` | `90` | Rolling buffer size in frames (~3 s @ 30 fps) |
| `BLINK_THRESHOLD` | `0.20` | EAR below this counts as a blink |
| `ENABLE_MOUTH_PENALTY` | `True` | Penalize wide-open mouths (mid-speech) |
| `POST_TRIGGER_SECONDS` | `1.0` | Extra buffering after SPACE (`0` = pure pre-trigger) |
| `AUTO_SNAP_THRESHOLD` | `0.26` | Minimum score for an auto-snap candidate |
| `AUTO_SNAP_STABLE_FRAMES` | `8` | Consecutive good frames before auto-firing |
| `AUTO_SNAP_COOLDOWN` | `5.0` | Seconds between auto-snaps |

### Opt-in advanced features (all default OFF)

| Feature | Enable via | Summary |
|---|---|---|
| **Adaptive calibration** | `python -m src.main --calibrate` or the GUI Calibrate button | Measures your personal resting EAR and derives a custom blink threshold, saved to `calibration.json` |
| **Composite scoring** | `USE_COMPOSITE_SCORING = True` | Blends eye/mouth score with blur (sharpness) and head-pose (front-facing) scores |
| **Adaptive learning** | `ENABLE_ADAPTIVE_LEARNING = True` | Keep/Discard buttons in the GUI nudge the auto-snap threshold based on your feedback over time |
| **Face weighting** | `USE_FACE_WEIGHTING = True` | Weights each face's contribution by size and centering, so a small background blinker doesn't ruin the shot |

---

## Testing

```bash
# Core unit tests — no webcam required
python tests/test_scoring.py
python tests/test_main.py
python tests/test_detector.py

# Feature tests
python tests/test_calibration.py
python tests/test_composite_scoring.py
python tests/test_feedback.py

# Integration test — opens the webcam
python tests/test_webcam.py
```

A benchmarking script (`tests/evaluate.py`) measures how closely the
algorithm's chosen frame matches a human-labelled ground truth across a set
of recorded clips — see `tests/eval_data/README.md` for dataset format and
usage.

---

## Packaging as a standalone executable

```bash
pip install pyinstaller

pyinstaller --onefile --windowed \
  --add-data "models/face_landmarker.task;models" \
  --hidden-import mediapipe \
  --hidden-import PIL._tkinter_finder \
  run.py
```

The bundled binary (in `dist/`) is typically 200–400 MB due to MediaPipe and
OpenCV. Test it on a machine without Python installed before sharing.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `opencv-python` | 4.13.0.92 | Webcam capture, image I/O, overlay drawing |
| `mediapipe` | 0.10.35 | Face Landmarker model (478 landmarks, Tasks API) |
| `numpy` | 2.5.0 | EAR / MAR geometry calculations |
| `Pillow` | ≥ 9.0 | BGR frame → Tkinter PhotoImage conversion (GUI only) |

Python 3.9–3.13 supported.

---

## License

Released under the [MIT License](LICENSE).
