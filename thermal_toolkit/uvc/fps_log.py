"""uvc / fps_log."""

from datetime import datetime
from pathlib import Path
import csv
import threading
import time


class UvcFpsCsvLogger:
    """Count received frames; write interval rates independently of capture/UI stalls."""

    def __init__(self, path, status_callback):
        self.path = Path(path)
        self._status = status_callback
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._count = 0
        self._last_count = 0
        self._started = self._last_sample = time.perf_counter()
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "elapsed_s", "interval_s", "received_frames", "fps", "total_frames"])
        self._file.flush()
        self._thread = threading.Thread(target=self._run, name="UVC FPS CSV", daemon=True)
        self._thread.start()

    def frame_received(self):
        with self._lock:
            if not self._stop.is_set():
                self._count += 1

    def _sample(self):
        with self._lock:
            now = time.perf_counter()
            count = self._count
        interval = now - self._last_sample
        if interval <= 0:
            return
        received = count - self._last_count
        self._writer.writerow([
            datetime.now().astimezone().isoformat(timespec="milliseconds"),
            f"{now - self._started:.6f}", f"{interval:.6f}", received,
            f"{received / interval:.6f}", count,
        ])
        self._last_sample = now
        self._last_count = count

    def _run(self):
        try:
            samples = 0
            while not self._stop.wait(1.0):
                self._sample()
                samples += 1
                if samples % 5 == 0:
                    self._file.flush()
            self._sample()  # Include the final partial interval.
            self._file.flush()
        except Exception as exc:
            self._status(f"FPS CSV write failed: {exc}; file: {self.path}")
        finally:
            self._file.close()
            self._status(f"FPS CSV closed: {self.path}")

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
