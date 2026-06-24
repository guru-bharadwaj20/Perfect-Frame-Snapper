"""
Webcam capture loop with a rolling frame buffer.

Maintains a deque of the last N frames so that when a "save" trigger fires,
the best frame from the recent past can be selected rather than only the
current frame.
"""

from collections import deque
import cv2


class FrameBuffer:
    """Opens the webcam and maintains a rolling buffer of recent frames."""

    def __init__(self, camera_index: int = 0, maxlen: int = 90):
        """
        Args:
            camera_index: OpenCV device index (0 = default webcam).
            maxlen: Maximum number of frames to keep (90 ≈ 3 s at 30 fps).
        """
        self.camera_index = camera_index
        self.buffer: deque = deque(maxlen=maxlen)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam; raise RuntimeError if the device can't be accessed."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Check that the webcam is connected and not in use by another app."
            )

    def read_frame(self) -> tuple[bool, object]:
        """Read a single frame and append it to the buffer.

        Returns:
            (ok, frame) where ok is False if the capture failed.
        """
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        ok, frame = self._cap.read()
        if ok:
            self.buffer.append(frame)
        return ok, frame

    def run(self, on_frame=None) -> None:
        """
        Blocking capture loop. Reads frames until the user presses 'q' or
        ESC, or until on_frame() returns False.

        Args:
            on_frame: Optional callable(frame) -> bool.  Return False to stop.
        """
        self.open()
        try:
            while True:
                ok, frame = self.read_frame()
                if not ok:
                    break
                if on_frame is not None and not on_frame(frame):
                    break
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            self.release()

    def release(self) -> None:
        """Release the webcam and destroy any OpenCV windows."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
