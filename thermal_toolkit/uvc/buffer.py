"""uvc / buffer."""

import threading
import time


class LatestFrameBuffer:
    """Thread-safe single-frame mailbox; newer frames replace pending old ones."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self._sequence = 0

    def submit(self, frame):
        if frame is None or not getattr(frame, "size", 0):
            return
        with self._lock:
            self._frame = frame.copy()
            self._frame_time = time.perf_counter()
            self._sequence += 1

    def take_latest(self, previous_sequence):
        with self._lock:
            if self._frame is None or self._sequence == previous_sequence:
                return None, previous_sequence, 0.0
            return self._frame.copy(), self._sequence, self._frame_time

    def clear(self):
        with self._lock:
            self._frame = None
            self._frame_time = 0.0
            self._sequence = 0
