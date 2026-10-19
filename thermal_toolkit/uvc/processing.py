"""uvc / processing."""

import numpy as np
import threading
import time
from thermal_toolkit.algorithms.mapping import raw_to_display_u8, uvc_to_gray
from thermal_toolkit.algorithms.pipeline import apply_algorithm_pipeline
from thermal_toolkit.core.qt import QThread, pyqtSignal


def process_uvc_pipeline_frame(
    frame,
    config,
    clahe_cache=None,
    fieldscale_states=None,
):
    started = time.perf_counter()
    original_gray = uvc_to_gray(frame)
    preprocessing_details = []
    pipeline_input_max = 255

    first_enabled = next(
        (stage for stage in config["stages"] if stage.enabled), None
    )
    use_native_fieldscale = (
        first_enabled is not None
        and first_enabled.algorithm_key == "fieldscale"
        and original_gray.dtype.itemsize > 1
    )
    if use_native_fieldscale:
        pipeline_input = original_gray.copy()
        pipeline_input_max = config["raw_max"]
        preprocessing_details.append("Native DN input for first Fieldscale step")
    elif original_gray.dtype == np.uint8:
        pipeline_input = original_gray.copy()
    else:
        pipeline_input = raw_to_display_u8(original_gray, config["raw_max"])

    final_image, records = apply_algorithm_pipeline(
        pipeline_input,
        config["stages"],
        input_max=pipeline_input_max,
        clahe_cache=clahe_cache,
        fieldscale_states=fieldscale_states,
        fieldscale_temporal=True,
    )
    total_ms = (time.perf_counter() - started) * 1000.0
    original_display = (
        original_gray.copy()
        if original_gray.dtype == np.uint8
        else raw_to_display_u8(original_gray, config["raw_max"])
    )
    return {
        "revision": config["revision"],
        "original_source": original_gray.copy(),
        "original_display": original_display,
        "pipeline_input": pipeline_input,
        "pipeline_input_max": pipeline_input_max,
        "final_image": final_image,
        "records": records,
        "preprocessing_details": preprocessing_details,
        "total_ms": total_ms,
    }


class UvcProcessingThread(QThread):
    frame_processed = pyqtSignal(object)
    frame_failed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending = None
        self._config = None
        self._config_revision = None
        self._stop_requested = False
        self._clahe_cache = {}
        self._fieldscale_states = {}

    def set_config(self, config):
        with self._condition:
            revision = None if config is None else config.get("revision")
            if revision != self._config_revision:
                self._clahe_cache.clear()
                self._fieldscale_states.clear()
                self._config_revision = revision
            self._config = config
            self._pending = None

    def submit_frame(self, frame):
        with self._condition:
            if self._stop_requested or self._config is None:
                return
            self._pending = (frame.copy(), self._config)
            self._condition.notify()

    def submit(self, frame, config):
        with self._condition:
            if self._stop_requested:
                return
            self._pending = (frame.copy(), config)
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stop_requested = True
            self._pending = None
            self._config = None
            self._condition.notify_all()

    def run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    return
                frame, config = self._pending
                self._pending = None
            try:
                payload = process_uvc_pipeline_frame(
                    frame,
                    config,
                    clahe_cache=self._clahe_cache,
                    fieldscale_states=self._fieldscale_states,
                )
            except Exception as exc:
                self.frame_failed.emit(
                    {"revision": config.get("revision"), "message": str(exc)}
                )
            else:
                self.frame_processed.emit(payload)
