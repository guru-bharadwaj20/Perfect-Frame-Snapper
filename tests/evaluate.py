"""
Evaluation / benchmarking for the Perfect-Frame scoring pipeline.

Loads pre-recorded frame sequences with human-labelled ground-truth best
frames, runs the scoring pipeline on every frame, and reports how well the
algorithm matches human judgment.

Usage
-----
    python tests/evaluate.py                           # default: tests/eval_data/
    python tests/evaluate.py --data path/to/clips
    python tests/evaluate.py --data tests/eval_data --labels labels.json --tol 30

Dataset layout
--------------
    tests/eval_data/
        labels.json          <- ground-truth file (see format below)
        clip_001/            <- one sub-folder per clip
            frame_0001.jpg
            frame_0002.jpg
            ...
        clip_002/
            ...

labels.json format
------------------
    [
      {
        "clip":       "clip_001",      <- must match the sub-folder name
        "best_frame": 47,              <- 1-based index of the human-labelled best frame
        "note":       "optional text"  <- ignored by the evaluator
      },
      ...
    ]

Metrics reported
----------------
  Exact match rate   fraction of clips where algo picks exactly the gt frame
  Near match rate    fraction within +/- tol frames of the gt frame
  Mean frame dist    average |algo_frame - gt_frame| in frames
  Mean score         average quality score at the algorithm's chosen frame
  Mean FPS           average frames scored per second (pipeline throughput)

Adding new clips
----------------
1.  Extract frames from your video as JPEG/PNG images (e.g. with ffmpeg):
        ffmpeg -i input.mp4 tests/eval_data/clip_XYZ/frame_%04d.jpg
2.  Watch the clip, identify the best frame index (1-based), and add an entry
    to tests/eval_data/labels.json.
3.  Re-run:  python tests/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import FaceDetector
from src.scoring import score_faces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_labels(labels_path: Path) -> list[dict]:
    with labels_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_frames(clip_dir: Path) -> list[Path]:
    """Return alphabetically-sorted image paths from a clip directory."""
    _EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        [p for p in clip_dir.iterdir() if p.suffix.lower() in _EXTS],
        key=lambda p: p.name,
    )


# ---------------------------------------------------------------------------
# Per-clip evaluation
# ---------------------------------------------------------------------------

def evaluate_clip(
    clip_dir: Path,
    gt_best_frame: int,          # 1-based
    detector: FaceDetector,
    use_composite: bool = False,
    use_face_weighting: bool = False,
) -> dict:
    """Run the scoring pipeline over all frames in clip_dir.

    Args:
        clip_dir:           Directory containing frame images.
        gt_best_frame:      Human-labelled best frame (1-based).
        detector:           An open FaceDetector instance.
        use_composite:      Pass True to enable composite scoring.
        use_face_weighting: Pass True to enable face prominence weighting.

    Returns:
        dict with evaluation results, or {"error": msg} on failure.
    """
    frames = _find_frames(clip_dir)
    if not frames:
        return {"error": f"No image files found in {clip_dir}"}

    best_score = -1.0
    best_idx   = 1
    total_ms   = 0.0
    scored     = 0

    for i, fpath in enumerate(frames, start=1):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        t0     = time.perf_counter()
        faces  = detector.detect(frame)
        result = score_faces(
            faces,
            frame=frame,
            use_composite=use_composite,
            use_face_weighting=use_face_weighting,
        )
        total_ms += (time.perf_counter() - t0) * 1000.0
        scored   += 1

        if result.score > best_score:
            best_score = result.score
            best_idx   = i

    if scored == 0:
        return {"error": f"No decodable images in {clip_dir}"}

    distance = abs(best_idx - gt_best_frame)
    avg_fps  = scored / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return {
        "clip":            clip_dir.name,
        "total_frames":    len(frames),
        "scored_frames":   scored,
        "gt_best_frame":   gt_best_frame,
        "algo_best_frame": best_idx,
        "distance":        distance,
        "algo_score":      round(float(best_score), 4),
        "avg_fps":         round(avg_fps, 1),
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    data_dir:           Path,
    labels_path:        Path,
    tol:                int  = 30,
    use_composite:      bool = False,
    use_face_weighting: bool = False,
) -> None:
    labels = _load_labels(labels_path)

    print(f"\nEvaluating {len(labels)} clip(s) from {data_dir}")
    print(f"  Tolerance  : +/-{tol} frames for near-match")
    print(f"  Composite  : {'ON' if use_composite else 'OFF'}")
    print(f"  Weighting  : {'ON' if use_face_weighting else 'OFF'}")
    print()

    results   = []
    distances = []
    scores    = []
    fps_vals  = []

    with FaceDetector() as detector:
        for entry in labels:
            clip_name = entry["clip"]
            gt_frame  = int(entry["best_frame"])
            clip_dir  = data_dir / clip_name

            if not clip_dir.is_dir():
                print(f"  SKIP  {clip_name:<22s} directory not found")
                continue

            r = evaluate_clip(
                clip_dir, gt_frame, detector,
                use_composite=use_composite,
                use_face_weighting=use_face_weighting,
            )
            if "error" in r:
                print(f"  ERROR {clip_name:<22s} {r['error']}")
                continue

            results.append(r)
            distances.append(r["distance"])
            scores.append(r["algo_score"])
            fps_vals.append(r["avg_fps"])

            if r["distance"] == 0:
                status = "EXACT"
            elif r["distance"] <= tol:
                status = f"NEAR  (+/-{r['distance']})"
            else:
                status = f"MISS  (+/-{r['distance']})"

            print(
                f"  {clip_name:<22s}  "
                f"gt={gt_frame:4d}  algo={r['algo_best_frame']:4d}  "
                f"score={r['algo_score']:.3f}  fps={r['avg_fps']:6.1f}  "
                f"[{status}]"
            )

    if not results:
        print("\nNo clips evaluated — check --data path and labels.json.")
        return

    n          = len(results)
    n_exact    = sum(1 for d in distances if d == 0)
    n_near     = sum(1 for d in distances if d <= tol)
    mean_dist  = sum(distances) / n
    mean_score = sum(scores) / n
    mean_fps   = sum(fps_vals) / n

    print()
    print("=" * 62)
    print(f"Results over {n} clip(s):")
    print(f"  Exact match rate   : {n_exact}/{n} = {n_exact/n*100:.1f}%")
    print(f"  Near match rate    : {n_near}/{n} = {n_near/n*100:.1f}%"
          f"  (tol={tol} frames)")
    print(f"  Mean frame distance: {mean_dist:.1f} frames")
    print(f"  Mean score chosen  : {mean_score:.3f}")
    print(f"  Mean scoring FPS   : {mean_fps:.1f} fps")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Perfect-Frame scoring pipeline against labelled clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", default="tests/eval_data",
        help="Directory containing clip sub-folders (default: tests/eval_data)",
    )
    parser.add_argument(
        "--labels", default="labels.json",
        help="Labels filename relative to --data dir (default: labels.json)",
    )
    parser.add_argument(
        "--tol", type=int, default=30,
        help="Frame-distance tolerance for 'near match' (default: 30)",
    )
    parser.add_argument(
        "--composite", action="store_true",
        help="Enable composite scoring (blur + pose) during evaluation",
    )
    parser.add_argument(
        "--face-weighting", action="store_true",
        help="Enable face prominence weighting during evaluation",
    )
    args = parser.parse_args()

    data_dir    = Path(args.data)
    labels_path = data_dir / args.labels

    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)
    if not labels_path.is_file():
        print(f"Error: labels file not found: {labels_path}")
        print(f"  Create {labels_path} — see the module docstring for the format.")
        sys.exit(1)

    run_evaluation(
        data_dir, labels_path,
        tol=args.tol,
        use_composite=args.composite,
        use_face_weighting=args.face_weighting,
    )


if __name__ == "__main__":
    main()
