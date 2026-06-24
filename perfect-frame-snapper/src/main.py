"""
Entry point and orchestrator for the Perfect-Frame Auto-Snapper.

STUB — Phase 4 implementation.

Planned flow:
    1. Instantiate FrameBuffer (capture.py) and start the webcam loop.
    2. On each frame, run FaceDetector (detector.py) to extract landmarks.
    3. Pass landmarks to score_frame (scoring.py) to evaluate frame quality.
    4. When a save trigger fires (keypress / periodic timer), walk the rolling
       buffer from newest to oldest, pick the frame with the highest score
       above the acceptance threshold, and write it to output/.
    5. Display a live preview with landmark overlays and the current score.

Usage (once implemented):
    python -m src.main
    python -m src.main --output-dir ./output --buffer-seconds 3
"""


def main() -> None:
    """Launch the Perfect-Frame Auto-Snapper."""
    raise NotImplementedError("main() will be implemented in Phase 4.")


if __name__ == "__main__":
    main()
