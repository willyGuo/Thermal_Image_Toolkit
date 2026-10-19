"""uvc / recording."""

from collections import deque
from pathlib import Path
import numpy as np
import threading
import time
from thermal_toolkit.core.constants import UVC_RECORDING_FOURCC, UVC_RECORDING_QUEUE_SIZE
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QThread,
)


class UvcRecordingThread(QThread):
    """Encode synchronized Original/Final comparison frames off the GUI thread."""

    def __init__(
        self,
        video_path,
        fps,
        parent=None,
        left_label="Original UVC",
        right_label="Final Pipeline Output",
    ):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.fps = max(1.0, float(fps))
        self.left_label = str(left_label)
        self.right_label = str(right_label)
        self._condition = threading.Condition()
        self._queue = deque()
        self._stop_requested = False
        self._stop_frame_time = None
        self._submitted_frames = 0
        self._queue_dropped_frames = 0
        self.result_summary = None
        self.error_message = None

    def submit(self, original, processed, frame_time=None):
        if original is None or processed is None:
            return False
        timestamp_value = (
            time.perf_counter() if frame_time is None else float(frame_time)
        )
        with self._condition:
            if self._stop_requested:
                return False
            if len(self._queue) >= UVC_RECORDING_QUEUE_SIZE:
                self._queue.popleft()
                self._queue_dropped_frames += 1
            self._queue.append(
                (original.copy(), processed.copy(), timestamp_value)
            )
            self._submitted_frames += 1
            self._condition.notify()
        return True

    def stop(self, frame_time=None):
        with self._condition:
            if not self._stop_requested:
                self._stop_frame_time = (
                    time.perf_counter()
                    if frame_time is None
                    else float(frame_time)
                )
            self._stop_requested = True
            self._condition.notify_all()

    @staticmethod
    def _display_bgr(image):
        array = np.asarray(image)
        if array.ndim != 2:
            raise ValueError("UVC recording requires 2D grayscale images.")
        if array.dtype == np.uint8:
            gray = np.ascontiguousarray(array)
        else:
            gray = np.clip(array, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @classmethod
    def _compose_frame(
        cls,
        original,
        processed,
        left_label="Original UVC",
        right_label="Final Pipeline Output",
    ):
        left = cls._display_bgr(original)
        right = cls._display_bgr(processed)
        if right.shape[:2] != left.shape[:2]:
            right = cv2.resize(
                right,
                (left.shape[1], left.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        canvas = np.ascontiguousarray(np.hstack((left, right)))
        if canvas.shape[0] % 2:
            canvas = cv2.copyMakeBorder(
                canvas, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        if canvas.shape[1] % 2:
            canvas = cv2.copyMakeBorder(
                canvas, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        panel_width = left.shape[1]
        cv2.rectangle(canvas, (0, 0), (panel_width - 1, 34), (0, 0, 0), -1)
        cv2.rectangle(
            canvas,
            (panel_width, 0),
            (canvas.shape[1] - 1, 34),
            (0, 0, 0),
            -1,
        )
        return cls._draw_comparison_labels(
            canvas, panel_width, left_label, right_label
        )

    @staticmethod
    def _draw_comparison_labels(canvas, panel_width, left_label, right_label):
        """Draw Unicode camera names into a BGR frame using Qt's system fonts."""
        # Paint on RGB32: Qt does not support painting on every BGR format.
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        ).convertToFormat(QImage.Format_RGB32)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        font = QFont()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(12, 24, str(left_label))
        painter.drawText(panel_width + 12, 24, str(right_label))
        painter.end()

        image = image.convertToFormat(QImage.Format_RGB888)
        bits = image.bits()
        bits.setsize(image.sizeInBytes())
        painted = np.frombuffer(bits, dtype=np.uint8).reshape(
            image.height(), image.bytesPerLine()
        )[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()
        return cv2.cvtColor(painted, cv2.COLOR_RGB2BGR)


    def run(self):
        writer = None
        first_frame_time = None
        last_frame_time = None
        last_canvas = None
        frame_size = None
        written_frames = 0
        duplicate_frames = 0
        timestamp_dropped_frames = 0

        try:
            if cv2 is None:
                raise RuntimeError(
                    "OpenCV not installed. pip install opencv-python"
                )
            while True:
                with self._condition:
                    while not self._queue and not self._stop_requested:
                        self._condition.wait()
                    if self._queue:
                        original, processed, frame_time = self._queue.popleft()
                    elif self._stop_requested:
                        break
                    else:
                        continue

                if first_frame_time is None:
                    first_frame_time = frame_time
                elapsed_seconds = max(0.0, frame_time - first_frame_time)
                canvas = self._compose_frame(
                    original,
                    processed,
                    self.left_label,
                    self.right_label,
                )
                current_size = (canvas.shape[1], canvas.shape[0])
                if writer is None:
                    frame_size = current_size
                    fourcc = cv2.VideoWriter_fourcc(*UVC_RECORDING_FOURCC)
                    writer = cv2.VideoWriter(
                        str(self.video_path), fourcc, self.fps, frame_size, True
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Cannot open UVC recording writer: {self.video_path}"
                        )
                elif current_size != frame_size:
                    raise RuntimeError(
                        "UVC recording frame size changed from "
                        f"{frame_size[0]}x{frame_size[1]} to "
                        f"{current_size[0]}x{current_size[1]}"
                    )

                target_index = int(round(elapsed_seconds * self.fps))
                if target_index < written_frames:
                    timestamp_dropped_frames += 1
                    last_frame_time = frame_time
                    continue
                while written_frames < target_index and last_canvas is not None:
                    writer.write(last_canvas)
                    written_frames += 1
                    duplicate_frames += 1
                writer.write(canvas)
                written_frames += 1
                last_canvas = canvas
                last_frame_time = frame_time

            if (
                writer is not None
                and last_canvas is not None
                and first_frame_time is not None
                and self._stop_frame_time is not None
            ):
                stop_elapsed = max(
                    0.0, self._stop_frame_time - first_frame_time
                )
                target_count = int(round(stop_elapsed * self.fps)) + 1
                while written_frames < target_count:
                    writer.write(last_canvas)
                    written_frames += 1
                    duplicate_frames += 1
        except Exception as exc:
            self.error_message = str(exc)
        finally:
            if writer is not None:
                writer.release()
            duration_seconds = 0.0
            if first_frame_time is not None:
                end_time = self._stop_frame_time
                if end_time is None:
                    end_time = last_frame_time
                if end_time is not None:
                    duration_seconds = max(0.0, end_time - first_frame_time)
            file_size = (
                self.video_path.stat().st_size
                if self.video_path.is_file()
                else 0
            )
            self.result_summary = {
                "codec": UVC_RECORDING_FOURCC,
                "fps": round(self.fps, 3),
                "frame_size": (
                    None
                    if frame_size is None
                    else {"width": frame_size[0], "height": frame_size[1]}
                ),
                "duration_seconds": round(duration_seconds, 3),
                "submitted_frames": self._submitted_frames,
                "written_frames": written_frames,
                "duplicate_frames": duplicate_frames,
                "queue_dropped_frames": self._queue_dropped_frames,
                "timestamp_dropped_frames": timestamp_dropped_frames,
                "file_size_bytes": int(file_size),
            }


class UvcViewerRecordingThread(UvcRecordingThread):
    """Encode one grayscale frame from Single UVC Statistic without text overlays."""

    def submit(self, frame, frame_time=None):
        if frame is None:
            return False
        timestamp_value = (
            time.perf_counter() if frame_time is None else float(frame_time)
        )
        with self._condition:
            if self._stop_requested:
                return False
            if len(self._queue) >= UVC_RECORDING_QUEUE_SIZE:
                self._queue.popleft()
                self._queue_dropped_frames += 1
            self._queue.append((frame.copy(), None, timestamp_value))
            self._submitted_frames += 1
            self._condition.notify()
        return True

    @classmethod
    def _compose_frame(
        cls,
        original,
        _processed=None,
        _left_label=None,
        _right_label=None,
    ):
        canvas = cls._display_bgr(original)
        if canvas.shape[0] % 2:
            canvas = cv2.copyMakeBorder(
                canvas, 0, 1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        if canvas.shape[1] % 2:
            canvas = cv2.copyMakeBorder(
                canvas, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return canvas
