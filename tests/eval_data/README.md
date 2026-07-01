# Evaluation Dataset

This folder holds labelled video clips for benchmarking the scoring pipeline via `tests/evaluate.py`.

## Current contents

| Clip | Frames | Best frame | Type |
|---|---|---|---|
| `clip_001` | 1 | 1 | Synthetic smoke-test (no faces) |
| `clip_002` | 15 | 8 | Synthetic blur-sharpness test |
| `clip_003` | 25 | 20 | Synthetic blur-sharpness test |
| `clip_004` | 10 | 3 | Synthetic blur-sharpness test |
| `clip_005` | 30 | 17 | Synthetic blur-sharpness test |

The 4 non-trivial clips (`clip_002`–`clip_005`) are meaningful only with composite scoring enabled.
Without `--composite`, all frames score 0.0 (no faces detected) and the algorithm always picks
frame 1, so the exact-match rate for these clips will be 0 % unless `--composite` is passed:

```
python tests/evaluate.py --composite    # uses blur component; matches ground truth
python tests/evaluate.py               # face-only scoring; 0.0 on synthetic data
```

## Dataset layout

```
tests/eval_data/
    labels.json              # ground-truth file (see format below)
    clip_001/
        frame_0001.jpg
    clip_002/
        frame_0001.jpg
        frame_0002.jpg
        ...
```

## labels.json format

```json
[
  {
    "clip": "clip_001",
    "best_frame": 1,
    "note": "optional human note"
  }
]
```

- `clip` must match the folder name under `eval_data/`
- `best_frame` is the **1-based** index of the best frame chosen by a human reviewer
- `note` is optional

## Adding real clips with real faces

Real clips with actual face content are required to evaluate EAR-based blink detection.

**Step 1 — extract frames from a video:**
```bash
ffmpeg -i recording.mp4 -q:v 2 tests/eval_data/clip_XYZ/frame_%04d.jpg
```

**Step 2 — watch the clip and pick the best frame:**
Play the clip and note the 1-based frame number where the subject has the most natural,
open-eyed, forward-facing expression.

**Step 3 — add to labels.json:**
```json
{ "clip": "clip_XYZ", "best_frame": 42, "note": "clearly eyes open, centred" }
```

**Step 4 — run the evaluator:**
```bash
python tests/evaluate.py --data tests/eval_data --labels tests/eval_data/labels.json
```

A dataset of 10–20 clips, each 50–150 frames, provides statistically meaningful accuracy numbers.
