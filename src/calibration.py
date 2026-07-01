"""
Per-user adaptive threshold calibration for Perfect-Frame Auto-Snapper.

CLI:   python -m src.main --calibrate
GUI:   click the "Calibrate" button in the right panel.

A 4-second guided capture collects open-eye EAR samples, computes a personal
baseline, and saves a personalised blink threshold to calibration.json.  On
next startup the app loads that file instead of the static BLINK_THRESHOLD.
Delete calibration.json to reset to config defaults.

Public API:
    data = run_calibration(detector, camera_index)
    save_calibration(path, data)
    data = load_calibration(path)          # None if absent / unreadable
    thr  = calibration_blink_threshold(data)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

_OPEN_EYE_SECONDS   = 4.0   # collect EAR samples for this many seconds
_THRESHOLD_FRACTION = 0.80  # personalised threshold = baseline * fraction
_MIN_SAMPLES        = 5     # fewest samples accepted before falling back


def run_calibration(detector, camera_index: int = 0) -> dict:
    """Guide user through a short open-eye capture and return calibration data.

    Opens a dedicated cv2 window so it works cleanly from the CLI.
    The GUI implements its own inline variant (see SnapperGUI._run_calibration_tick).

    Args:
        detector:     An already-opened FaceDetector instance.
        camera_index: Webcam device index.

    Returns:
        dict with keys 'baseline_ear', 'blink_threshold', 'sample_count'.
        Falls back to config defaults when no face was detected.
    """
    import cv2
    from src.capture import FrameBuffer
    from src.scoring import eye_aspect_ratio
    from src.config import BLINK_THRESHOLD

    buf = FrameBuffer(camera_index=camera_index, maxlen=1)
    buf.open()

    ear_samples: list[float] = []
    start = time.monotonic()

    print(f"\n[Calibration] Keep eyes NATURALLY OPEN for {_OPEN_EYE_SECONDS:.0f} s …")
    try:
        while True:
            ok, frame = buf.read_frame()
            if not ok:
                break

            elapsed   = time.monotonic() - start
            remaining = _OPEN_EYE_SECONDS - elapsed

            vis   = frame.copy()
            h, w  = vis.shape[:2]

            if remaining > 0:
                msg = f"Calibrating — keep eyes open  ({remaining:.1f}s)"
                clr = (200, 200, 0)
            else:
                msg = "Done!  Saving calibration …"
                clr = (0, 220, 0)

            cv2.putText(vis, msg, (10, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, clr, 2, cv2.LINE_AA)
            cv2.putText(vis,
                        "Look at the camera — blink naturally only AFTER this phase",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (160, 160, 160), 1, cv2.LINE_AA)

            faces = detector.detect(frame)
            if faces:
                for face in faces:
                    r   = eye_aspect_ratio(face.right_eye_pts)
                    l   = eye_aspect_ratio(face.left_eye_pts)
                    avg = (r + l) * 0.5
                    cv2.putText(vis, f"EAR: {avg:.3f}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    if remaining > 0:
                        ear_samples.append(avg)
            else:
                cv2.putText(vis, "No face detected — look at camera",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 100, 255), 1, cv2.LINE_AA)

            cv2.imshow("Perfect-Frame Calibration", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if elapsed >= _OPEN_EYE_SECONDS:
                break
    finally:
        try:
            cv2.destroyWindow("Perfect-Frame Calibration")
        except Exception:
            pass
        buf.release()

    return _compute_calibration(ear_samples)


def _compute_calibration(ear_samples: list[float]) -> dict:
    """Compute calibration result from collected EAR samples."""
    from src.config import BLINK_THRESHOLD

    if len(ear_samples) < _MIN_SAMPLES:
        print(f"[Calibration] Too few samples ({len(ear_samples)}) — using config default.")
        return {
            "baseline_ear":    round(float(BLINK_THRESHOLD) / _THRESHOLD_FRACTION, 4),
            "blink_threshold": float(BLINK_THRESHOLD),
            "sample_count":    len(ear_samples),
        }

    arr      = np.array(ear_samples, dtype=np.float64)
    filtered = arr[arr >= np.percentile(arr, 10)]   # drop lowest 10 %
    baseline  = float(np.median(filtered))
    threshold = round(baseline * _THRESHOLD_FRACTION, 4)

    print(f"[Calibration] baseline EAR={baseline:.4f}  "
          f"-> personalised threshold={threshold:.4f}  "
          f"(samples={len(ear_samples)})")
    return {
        "baseline_ear":    round(baseline, 4),
        "blink_threshold": threshold,
        "sample_count":    len(ear_samples),
    }


def compute_calibration_from_samples(ear_samples: list[float]) -> dict:
    """Public entry point used by the GUI's inline calibration."""
    return _compute_calibration(ear_samples)


def save_calibration(path: Path, data: dict) -> None:
    """Persist calibration data as JSON."""
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[Calibration] Saved -> {path}")


def load_calibration(path: Path) -> Optional[dict]:
    """Load calibration data.  Returns None if absent or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def calibration_blink_threshold(data: dict) -> float:
    """Extract the personalised blink threshold from calibration data.

    Falls back to config.BLINK_THRESHOLD if the key is missing.
    """
    from src.config import BLINK_THRESHOLD
    return float(data.get("blink_threshold", BLINK_THRESHOLD))
