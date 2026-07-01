"""
Lightweight adaptive-learning feedback loop for Perfect-Frame Auto-Snapper.

When ENABLE_ADAPTIVE_LEARNING=True the GUI shows "Keep / Discard" buttons
after each snap.  Each decision is appended to feedback_log.json.  Every
ADAPTIVE_LEARNING_WINDOW decisions, the auto-snap threshold is nudged in
memory (never written back to config.py).

Heuristic:
  acceptance_rate < 0.50  →  raise threshold by step  (photos too often bad)
  acceptance_rate > 0.80  →  lower threshold by step  (missing good moments)
  otherwise               →  no change

The adjusted threshold is clamped to [0.15, 0.45].

Public API:
    log_decision(path, score, kept)
    compute_adjusted_threshold(path, base, window, step) -> float
    decision_count(path) -> int
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def log_decision(path: Path, score: float, kept: bool) -> None:
    """Append one keep/discard decision to the JSON feedback log.

    Args:
        path:  Path to the JSON log file (created if absent).
        score: Frame quality score at decision time.
        kept:  True = user kept the photo; False = user discarded it.
    """
    path = Path(path)
    decisions: list[dict] = []
    if path.exists():
        try:
            decisions = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(decisions, list):
                decisions = []
        except Exception:
            decisions = []

    decisions.append({
        "timestamp": round(time.time(), 3),
        "score":     round(float(score), 4),
        "kept":      bool(kept),
    })
    path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")


def decision_count(path: Path) -> int:
    """Return the total number of logged decisions."""
    path = Path(path)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def compute_adjusted_threshold(
    path: Path,
    base_threshold: float,
    window: int = 20,
    step: float = 0.01,
) -> float:
    """Compute the adjusted AUTO_SNAP_THRESHOLD from all logged decisions.

    Processes every complete *window* of decisions in chronological order,
    accumulating threshold adjustments.  The result is clamped to [0.15, 0.45].

    Args:
        path:           Path to the feedback log JSON.
        base_threshold: Starting threshold (usually config.AUTO_SNAP_THRESHOLD).
        window:         Decisions per adjustment cycle.
        step:           Threshold delta per cycle.

    Returns:
        Adjusted threshold float (may equal base_threshold if < window decisions
        have been logged or the acceptance rate is in the neutral band).
    """
    path = Path(path)
    if not path.exists():
        return base_threshold

    try:
        decisions = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decisions, list):
            return base_threshold
    except Exception:
        return base_threshold

    threshold    = base_threshold
    n_complete   = len(decisions) // window

    for i in range(n_complete):
        chunk      = decisions[i * window: (i + 1) * window]
        kept_count = sum(1 for d in chunk if d.get("kept", False))
        rate       = kept_count / window

        if rate < 0.50:
            threshold += step      # too many discards → be pickier
        elif rate > 0.80:
            threshold -= step      # almost always kept → be more relaxed

    return float(max(0.15, min(0.45, threshold)))
