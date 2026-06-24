# Perfect-Frame Auto-Snapper

Continuously buffers the last few seconds of webcam footage and retroactively
saves the single best frame — the one where everyone in shot has their eyes
fully open, no blinks, no mid-word expressions.

## How it works

A rolling deque keeps the last ~3 seconds of frames in memory.  When a save
trigger fires, the tool scores every buffered frame using the Eye Aspect Ratio
(EAR) derived from MediaPipe Face Mesh landmarks and writes the highest-scoring
frame to `output/`.

## Planned phases

| Phase | Description |
|-------|-------------|
| 1 | Project scaffold, webcam capture, rolling buffer, smoke test |
| 2 | MediaPipe face detection + landmark extraction (`detector.py`) |
| 3 | EAR computation + frame scoring (`scoring.py`) |
| 4 | Save trigger, best-frame selection, live preview overlay (`main.py`) |
| 5 | Tuning, CLI flags, packaging |

## Requirements

- Python 3.9–3.13
- Webcam accessible at device index 0

## Setup

```bash
# Create and activate the virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the Phase 1 smoke test

```bash
python tests/test_webcam.py
```

Expected output:

```
Webcam opened. Capturing 30 frames...
  Frames captured : 30
  Frame dimensions: 1280 x 720 px, 3 channels
  Buffer length   : 30
PASS: webcam smoke test succeeded.
```

## Project structure

```
perfect-frame-snapper/
├── src/
│   ├── capture.py      # Webcam loop + rolling frame buffer
│   ├── detector.py     # Face detection + landmarks (stub → Phase 2)
│   ├── scoring.py      # EAR + frame scoring (stub → Phase 3)
│   └── main.py         # Entry point / orchestrator (stub → Phase 4)
├── output/             # Saved best-frame JPEGs
├── tests/
│   └── test_webcam.py  # Smoke test
└── requirements.txt
```
