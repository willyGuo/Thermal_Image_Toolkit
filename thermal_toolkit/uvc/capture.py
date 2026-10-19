"""uvc / capture."""

import sys
import threading
import time
from thermal_toolkit.core.constants import (
    DEFAULT_UVC_CAPTURE_MODE,
    UVC_AUTO_DETECT_FRAME_TIMEOUT_SECONDS,
    UVC_CAPTURE_FOURCC,
    UVC_CAPTURE_MODE_OPTIONS,
)
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import QThread, pyqtSignal


class UvcCaptureThread(QThread):
    frame_ready = pyqtSignal(object)
    frame_seen = pyqtSignal(float)
    camera_opened = pyqtSignal(int, int, int, float)
    status_changed = pyqtSignal(str)
    stream_interrupted = pyqtSignal(str)

    def __init__(
        self,
        camera_index,
        width,
        height,
        fps,
        reconnect_seconds,
        parent=None,
        frame_consumer=None,
        capture_mode=DEFAULT_UVC_CAPTURE_MODE,
        camera_device_id=None,
        camera_name=None,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_seconds = reconnect_seconds
        self.frame_consumer = frame_consumer
        self.capture_mode = capture_mode
        self.camera_device_id = camera_device_id
        self.camera_name = camera_name or f"Camera {camera_index}"
        self.selected_capture_mode = None
        self.fps_logger = None
        self._camera_info_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reconnect_event = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture = None

    def stop(self):
        self._stop_event.set()
        if self.fps_logger is not None:
            self.fps_logger.close()

    def force_reconnect(self):
        self._reconnect_event.set()

    def update_camera_device(self, camera_index, camera_name=None):
        with self._camera_info_lock:
            changed = self.camera_index != camera_index
            self.camera_index = camera_index
            if camera_name:
                self.camera_name = camera_name
        if changed:
            self._reconnect_event.set()

    def _camera_device_snapshot(self):
        with self._camera_info_lock:
            return self.camera_index, self.camera_name

    def _set_capture(self, capture):
        with self._capture_lock:
            self._capture = capture

    def _clear_capture(self, capture):
        with self._capture_lock:
            if self._capture is capture:
                self._capture = None
        try:
            capture.release()
        except Exception:
            pass

    def _open_capture(self, capture_mode=None, camera_index=None):
        capture_mode = capture_mode or self.capture_mode
        if camera_index is None:
            camera_index, _camera_name = self._camera_device_snapshot()
        if camera_index is None:
            return None
        if capture_mode == "generic_auto":
            if sys.platform.startswith("win"):
                backends = [cv2.CAP_DSHOW]
                if hasattr(cv2, "CAP_MSMF"):
                    backends.append(cv2.CAP_MSMF)
                for backend in backends:
                    try:
                        capture = cv2.VideoCapture(camera_index, backend)
                    except (TypeError, cv2.error):
                        continue
                    if capture.isOpened():
                        return capture
                    capture.release()
            return cv2.VideoCapture(camera_index)

        if sys.platform.startswith("win"):
            mjpg_fourcc = cv2.VideoWriter_fourcc(*UVC_CAPTURE_FOURCC)
            open_params = [
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width,
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height,
                cv2.CAP_PROP_FOURCC,
                mjpg_fourcc,
            ]
            try:
                # DirectShow applies these values before starting the device.
                # Setting CAP_PROP_FPS afterward can reopen it without FOURCC
                # and make the Novatek host request format=0 instead of MJPEG.
                capture = cv2.VideoCapture(
                    camera_index,
                    cv2.CAP_DSHOW,
                    open_params,
                )
            except (TypeError, cv2.error):
                capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

            if capture.isOpened():
                return capture

            capture.release()
            capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                capture.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
            return capture

        capture = cv2.VideoCapture(camera_index)
        if capture.isOpened():
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*UVC_CAPTURE_FOURCC),
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
        return capture

    @staticmethod
    def _capture_mode_name(capture_mode):
        for mode_name, mode_key in UVC_CAPTURE_MODE_OPTIONS:
            if mode_key == capture_mode:
                return mode_name
        return capture_mode

    @staticmethod
    def _alternate_auto_mode(capture_mode):
        if capture_mode == "novatek_mjpg":
            return "generic_auto"
        return "novatek_mjpg"

    @staticmethod
    def _fourcc_text(value):
        fourcc = int(value)
        text = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        return text.strip("\x00 ") or "UNKNOWN"

    def run(self):
        capture = None
        next_auto_mode = "novatek_mjpg"
        stream_started = False
        try:
            while not self._stop_event.is_set():
                self._reconnect_event.clear()
                camera_index, camera_name = self._camera_device_snapshot()
                if camera_index is None:
                    if stream_started:
                        message = (
                            f"{camera_name} disconnected after streaming "
                            "started; stopping camera."
                        )
                        self.status_changed.emit(message)
                        self.stream_interrupted.emit(message)
                        return
                    self.status_changed.emit(
                        f"{camera_name} disconnected; waiting for hot-plug..."
                    )
                    if self._stop_event.wait(1.0):
                        break
                    continue
                active_capture_mode = self.capture_mode
                if self.capture_mode == "auto_detect":
                    active_capture_mode = next_auto_mode
                    self.status_changed.emit(
                        f"Auto Detect: trying {camera_name} "
                        f"(index {camera_index}) with "
                        f"{self._capture_mode_name(active_capture_mode)}..."
                    )
                else:
                    self.status_changed.emit(
                        f"Opening {camera_name} (index {camera_index}) "
                        f"({self._capture_mode_name(active_capture_mode)})..."
                    )
                try:
                    capture = self._open_capture(
                        active_capture_mode, camera_index
                    )
                except Exception as exc:
                    capture = None
                    self.status_changed.emit(f"Camera open error: {exc}; retrying...")

                if capture is None or not capture.isOpened():
                    if capture is not None:
                        self._clear_capture(capture)
                        capture = None
                    if stream_started:
                        message = (
                            f"Cannot reopen {camera_name} after streaming "
                            "was interrupted; stopping camera."
                        )
                        self.status_changed.emit(message)
                        self.stream_interrupted.emit(message)
                        return
                    if self.capture_mode == "auto_detect":
                        next_auto_mode = self._alternate_auto_mode(
                            active_capture_mode
                        )
                    self.status_changed.emit(
                        f"Cannot open {camera_name} (index {camera_index}); "
                        "retrying..."
                    )
                    if self._stop_event.wait(1.0):
                        break
                    continue

                self._set_capture(capture)
                if not sys.platform.startswith("win") and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not sys.platform.startswith("win") and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
                actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
                actual_fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
                actual_fourcc = self._fourcc_text(actual_fourcc_value)
                if (
                    sys.platform.startswith("win")
                    and active_capture_mode == "novatek_mjpg"
                    and actual_fourcc != UVC_CAPTURE_FOURCC
                ):
                    self.status_changed.emit(
                        f"{camera_name} negotiated {actual_fourcc!r}, "
                        f"expected {UVC_CAPTURE_FOURCC}; retrying..."
                    )
                    self._clear_capture(capture)
                    capture = None
                    if self.capture_mode == "auto_detect":
                        next_auto_mode = "generic_auto"
                    if self._stop_event.wait(1.0):
                        break
                    continue
                self.status_changed.emit(
                    f"{camera_name} (index {camera_index}): {actual_fourcc} "
                    f"{actual_width}x{actual_height} {actual_fps:.2f} FPS"
                )
                self.camera_opened.emit(
                    camera_index, actual_width, actual_height, actual_fps
                )

                failed_since = None
                received_frame = False
                last_frame_seen_emit = 0.0
                while not self._stop_event.is_set():
                    try:
                        ok, frame = capture.read()
                    except Exception:
                        ok, frame = False, None

                    if self._stop_event.is_set():
                        break
                    if self._reconnect_event.is_set():
                        if stream_started:
                            message = (
                                f"{camera_name} stream interrupted; "
                                "stopping camera."
                            )
                            self.status_changed.emit(message)
                            self.stream_interrupted.emit(message)
                            return
                        self.status_changed.emit(
                            f"Reconnecting {camera_name} (index {camera_index})..."
                        )
                        break
                    if ok and frame is not None and frame.size:
                        if self.fps_logger is not None:
                            self.fps_logger.frame_received()
                        failed_since = None
                        stream_started = True
                        if not received_frame:
                            received_frame = True
                            self.selected_capture_mode = active_capture_mode
                            if self.capture_mode == "auto_detect":
                                next_auto_mode = active_capture_mode
                                self.status_changed.emit(
                                    f"Auto Detect selected: "
                                    f"{self._capture_mode_name(active_capture_mode)}; "
                                    f"{actual_fourcc} {actual_width}x{actual_height} "
                                    f"{actual_fps:.2f} FPS"
                                )
                        now = time.perf_counter()
                        if now - last_frame_seen_emit >= 0.1:
                            self.frame_seen.emit(now)
                            last_frame_seen_emit = now
                        if self.frame_consumer is None:
                            self.frame_ready.emit(frame)
                        else:
                            self.frame_consumer(frame)
                        continue

                    now = time.perf_counter()
                    if failed_since is None:
                        failed_since = now
                    no_frame_timeout = self.reconnect_seconds
                    if self.capture_mode == "auto_detect" and not received_frame:
                        no_frame_timeout = UVC_AUTO_DETECT_FRAME_TIMEOUT_SECONDS
                    if now - failed_since >= no_frame_timeout:
                        if stream_started:
                            message = (
                                f"{camera_name} stream interrupted for "
                                f"{no_frame_timeout:.1f}s; stopping camera."
                            )
                            self.status_changed.emit(message)
                            self.stream_interrupted.emit(message)
                            return
                        if self.capture_mode == "auto_detect" and not received_frame:
                            next_auto_mode = self._alternate_auto_mode(
                                active_capture_mode
                            )
                        self.status_changed.emit(
                            f"No frame for {no_frame_timeout:.1f}s; "
                            f"reconnecting {camera_name} "
                            f"(index {camera_index})..."
                        )
                        break
                    if self._stop_event.wait(0.03):
                        break

                self._clear_capture(capture)
                capture = None
                if not self._stop_event.is_set():
                    self._stop_event.wait(0.25)
        finally:
            if self.fps_logger is not None:
                self.fps_logger.close()
            if capture is not None:
                self._clear_capture(capture)
