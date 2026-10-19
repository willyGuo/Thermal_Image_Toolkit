"""features / uvc_algorithm / page."""
import json
import numpy as np
import time
from thermal_toolkit.algorithms.pipeline import pipeline_execution_json, pipeline_record_image
from thermal_toolkit.core.constants import DEFAULT_UVC_CAPTURE_MODE, DEFAULT_UVC_FPS, UVC_FRAME_WAIT_SECONDS, UVC_RECONNECT_SECONDS
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import QGridLayout, QLabel, QMessageBox, QPushButton, QSizePolicy, QTimer, QWidget
from thermal_toolkit.io.export import save_png
from thermal_toolkit.reports.pipeline import save_pipeline_overview_png, save_uvc_comparison_png
from thermal_toolkit.ui.pipeline_editor import AlgorithmPipelineEditor
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox
from thermal_toolkit.uvc.capture import UvcCaptureThread
from thermal_toolkit.uvc.processing import UvcProcessingThread, process_uvc_pipeline_frame

class UvcAlgorithmPageMixin:
    """features / uvc_algorithm / page behavior composed by the main window."""

    def build_uvc_algorithm_page(self):
        p, f = self._form_page()
        self.uvc_camera_combo = FocusWheelComboBox()
        self.uvc_pipeline_editor = AlgorithmPipelineEditor(allow_step_algorithm_change=False, fieldscale_temporal=True)
        self.uvc_preview_stage_combo = FocusWheelComboBox()
        self.uvc_preview_stage_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.uvc_camera_combo.currentIndexChanged.connect(self.switch_uvc_camera)
        self.bit_depth_combo.currentIndexChanged.connect(self.mark_uvc_processing_config_changed)
        self.uvc_pipeline_editor.pipeline_changed.connect(self.update_uvc_pipeline_ui)
        self.uvc_preview_stage_combo.currentIndexChanged.connect(self.update_uvc_stage_preview)
        controls = QWidget()
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.uvc_start_btn = QPushButton('Start Camera')
        self.uvc_start_btn.setObjectName('startButton')
        self.uvc_start_btn.clicked.connect(self.start_uvc_camera)
        self.uvc_stop_btn = QPushButton('Stop Camera')
        self.uvc_stop_btn.setObjectName('stopButton')
        self.uvc_stop_btn.clicked.connect(self.stop_uvc_camera)
        self.uvc_stop_btn.setEnabled(False)
        self.uvc_capture_btn = QPushButton('Capture Pair')
        self.uvc_capture_btn.setObjectName('captureButton')
        self.uvc_capture_btn.clicked.connect(self.capture_uvc_pair)
        self.uvc_capture_btn.setEnabled(False)
        self.uvc_record_start_btn = QPushButton('Start Recording')
        self.uvc_record_start_btn.setObjectName('startButton')
        self.uvc_record_start_btn.setToolTip('Record synchronized Original UVC and Final Pipeline Output side by side as MJPG AVI')
        self.uvc_record_start_btn.clicked.connect(self.start_uvc_recording)
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn = QPushButton('Stop Recording')
        self.uvc_record_stop_btn.setObjectName('stopButton')
        self.uvc_record_stop_btn.clicked.connect(self.stop_uvc_recording)
        self.uvc_record_stop_btn.setEnabled(False)
        self.uvc_refresh_btn = QPushButton('Refresh Cameras')
        self.uvc_refresh_btn.clicked.connect(self.refresh_camera_devices)
        controls_layout.addWidget(self.uvc_start_btn, 0, 0)
        controls_layout.addWidget(self.uvc_stop_btn, 0, 1)
        controls_layout.addWidget(self.uvc_capture_btn, 1, 0, 1, 2)
        controls_layout.addWidget(self.uvc_record_start_btn, 2, 0)
        controls_layout.addWidget(self.uvc_record_stop_btn, 2, 1)
        controls_layout.addWidget(self.uvc_refresh_btn, 3, 0, 1, 2)
        self.uvc_status = QLabel('Camera stopped')
        self.uvc_status.setWordWrap(True)
        self.uvc_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow('CAMERA_INDEX', self.uvc_camera_combo)
        f.addRow(controls)
        f.addRow('Status', self.uvc_status)
        f.addRow(self.uvc_pipeline_editor)
        f.addRow('Right Preview', self.uvc_preview_stage_combo)
        self.stack.addWidget(p)
        self.update_uvc_pipeline_ui()

    def uvc_result_title(self):
        return self.uvc_preview_stage_combo.currentText()

    def mark_uvc_processing_config_changed(self, _value=None):
        if self.uvc_recording_worker is not None:
            self.stop_uvc_recording(reason='Pipeline configuration changed')
        self.uvc_processing_revision += 1
        if hasattr(self, 'uvc_capture_btn'):
            self.uvc_capture_btn.setEnabled(False)
        worker = self.uvc_processing_worker
        if worker is not None and worker.isRunning():
            try:
                config = self.build_uvc_processing_config()
            except Exception as exc:
                worker.set_config(None)
                self.uvc_status.setText(f'Pipeline configuration error: {exc}')
            else:
                worker.set_config(config)

    def update_uvc_pipeline_ui(self, _value=None):
        self.rebuild_pipeline_preview_combo(self.uvc_pipeline_editor, self.uvc_preview_stage_combo)
        self.mark_uvc_processing_config_changed()
        self.update_uvc_result_title()
        self.refresh_uvc_processed_preview()

    def update_uvc_result_title(self):
        if self.current_function_index() == len(self.FUNCTIONS) - 1:
            title = self.uvc_result_title()
            self.preview_result.set_title(title)
            self.preview_result_caption.setText(title)

    def build_uvc_processing_config(self):
        stages = self.uvc_pipeline_editor.stages()
        if not any((stage.enabled for stage in stages)):
            raise ValueError('Add and enable at least one UVC algorithm step.')
        return {'revision': self.uvc_processing_revision, 'capture_mode': DEFAULT_UVC_CAPTURE_MODE, 'raw_max': self.raw_max, 'stages': stages}

    def refresh_uvc_processed_preview(self):
        worker = self.uvc_processing_worker
        if self.uvc_original_source is None or worker is None or (not worker.isRunning()):
            return
        try:
            config = self.build_uvc_processing_config()
        except Exception as exc:
            message = str(exc)
            if message != self.uvc_last_processing_error:
                self.log.append(f'UVC frame processing error: {message}')
            self.uvc_last_processing_error = message
            self.uvc_processed_frame = None
            self.current_result = None
            self.preview_result.set_array(None)
            self.uvc_capture_btn.setEnabled(False)
            self.uvc_status.setText(f'Frame processing error: {message}')
            return
        worker.submit(self.uvc_original_source, config)

    def update_uvc_stage_preview(self, _combo_index=None):
        if self.uvc_processed_frame is None:
            self.update_uvc_result_title()
            return
        selected_pipeline_index = self.uvc_preview_stage_combo.currentData()
        preview = pipeline_record_image(self.uvc_pipeline_records, selected_pipeline_index, self.uvc_processed_frame)
        self.current_result = preview
        if self.current_function_index() == len(self.FUNCTIONS) - 1:
            self.preview_result.set_array(preview)
        self.update_uvc_result_title()

    def process_uvc_frame(self, original_gray):
        return process_uvc_pipeline_frame(original_gray, self.build_uvc_processing_config(), clahe_cache={}, fieldscale_states={})

    def build_uvc_overview_items(self):
        input_label = 'Native DN' if self.uvc_pipeline_input_max > 255 else 'UVC Gray'
        items = [('00 Original UVC', self.uvc_original_frame, 255), (f'01 Pipeline Input - {input_label}', self.uvc_pipeline_input, self.uvc_pipeline_input_max)]
        for overview_index, record in enumerate(self.uvc_pipeline_records, start=2):
            items.append((f"{overview_index:02d} Step {record['pipeline_index']:02d} {record['name']}", record['image'], 255))
        return items

    def start_uvc_camera(self, _checked=False):
        if cv2 is None:
            message = 'OpenCV not installed. pip install opencv-python'
            self.log.append(f'\nERROR: {message}')
            QMessageBox.critical(self, 'Camera Error', message)
            return
        try:
            processing_config = self.build_uvc_processing_config()
        except Exception as exc:
            message = str(exc)
            self.log.append(f'\nERROR: {message}')
            QMessageBox.critical(self, 'UVC Pipeline Error', message)
            return
        if not self.stop_uvc_camera(log_message=False):
            self.uvc_restart_when_stopped = True
            self.log.append('Previous capture thread is still stopping; camera will start automatically afterward.')
            return
        self.uvc_restart_when_stopped = False
        capture_mode = DEFAULT_UVC_CAPTURE_MODE
        camera_info = self.selected_camera_info(self.uvc_camera_combo)
        camera_index = camera_info['index']
        if camera_index is None:
            QMessageBox.warning(self, 'Camera Disconnected', 'Select a connected camera before starting UVC.')
            return
        requested_fps = DEFAULT_UVC_FPS
        processing_worker = UvcProcessingThread(self)
        processing_worker.set_config(processing_config)
        processing_worker.frame_processed.connect(self.handle_uvc_processed_frame)
        processing_worker.frame_failed.connect(self.handle_uvc_processing_failed)
        processing_worker.finished.connect(processing_worker.deleteLater)
        worker = UvcCaptureThread(camera_index, self.width, self.height, requested_fps, UVC_RECONNECT_SECONDS, self, frame_consumer=processing_worker.submit_frame, capture_mode=capture_mode, camera_device_id=camera_info['device_id'], camera_name=camera_info['name'])
        worker.frame_seen.connect(self.handle_uvc_frame_seen)
        worker.camera_opened.connect(self.handle_uvc_camera_opened)
        worker.status_changed.connect(self.handle_uvc_worker_status)
        worker.stream_interrupted.connect(self.handle_uvc_stream_interrupted)
        worker.finished.connect(self.handle_uvc_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self.uvc_worker = worker
        self.uvc_processing_worker = processing_worker
        self.uvc_original_source = None
        self.uvc_original_frame = None
        self.uvc_processed_frame = None
        self.uvc_pipeline_input = None
        self.uvc_pipeline_input_max = 255
        self.uvc_pipeline_records = []
        self.uvc_pipeline_total_ms = 0.0
        self.uvc_last_processing_error = None
        self.uvc_frame_count = 0
        self.uvc_fps_display = 0.0
        now = time.perf_counter()
        self.uvc_last_time = now
        self.uvc_last_frame_time = now
        self.uvc_waiting_for_frame = False
        self.uvc_reconnect_requested = False
        self.uvc_actual_fps = 0.0
        self.preview_original.set_array(None)
        self.preview_result.set_array(None)
        self.uvc_watchdog_timer.start()
        self.uvc_start_btn.setEnabled(False)
        self.uvc_stop_btn.setEnabled(True)
        self.uvc_capture_btn.setEnabled(False)
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn.setEnabled(False)
        self.uvc_status.setText(f"Opening {camera_info['name']} (Camera {camera_index}, {capture_mode})...")
        self.log.append(f"Starting {camera_info['name']} (Camera {camera_index}) in {capture_mode} mode using the background capture thread...")
        processing_order = self.uvc_pipeline_editor.summary()
        self.log.append(f'UVC processing order: {processing_order}')
        self.log.append('UVC processor uses latest-frame-wins; pending old frames are replaced.')
        processing_worker.start()
        worker.start()

    def stop_uvc_processing_worker(self, wait_ms=5000):
        worker = self.uvc_processing_worker
        if worker is None:
            return True
        worker.stop()
        if worker.isRunning() and (not worker.wait(wait_ms)):
            return False
        self.uvc_processing_worker = None
        return True

    def stop_uvc_camera(self, _checked=False, log_message=True):
        self.uvc_restart_when_stopped = False
        self.close_uvc_video_window()
        self.stop_uvc_recording(reason='Camera stopped', log_message=log_message)
        worker = self.uvc_worker
        was_running = worker is not None or self.uvc_processing_worker is not None
        self.uvc_watchdog_timer.stop()
        if worker is not None:
            worker.stop()
            if not worker.wait(2000):
                self.uvc_start_btn.setEnabled(False)
                self.uvc_stop_btn.setEnabled(False)
                self.uvc_capture_btn.setEnabled(False)
                self.uvc_record_start_btn.setEnabled(False)
                self.uvc_record_stop_btn.setEnabled(False)
                self.uvc_status.setText('Camera capture thread is stopping...')
                return False
            self.uvc_worker = None
        if not self.stop_uvc_processing_worker():
            self.uvc_start_btn.setEnabled(False)
            self.uvc_stop_btn.setEnabled(False)
            self.uvc_capture_btn.setEnabled(False)
            self.uvc_record_start_btn.setEnabled(False)
            self.uvc_record_stop_btn.setEnabled(False)
            self.uvc_status.setText('UVC processing thread is stopping...')
            return False
        self.uvc_start_btn.setEnabled(True)
        self.uvc_stop_btn.setEnabled(False)
        self.uvc_capture_btn.setEnabled(False)
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn.setEnabled(False)
        self.uvc_actual_fps = 0.0
        self.uvc_status.setText('Camera stopped')
        if was_running and log_message:
            self.log.append('Camera stopped')
        return True

    def switch_uvc_camera(self, _combo_index):
        if self.uvc_worker is not None:
            self.start_uvc_camera()

    def handle_uvc_camera_opened(self, camera_index, actual_width, actual_height, actual_fps):
        if self.sender() is not self.uvc_worker:
            return
        self.uvc_actual_fps = float(actual_fps)
        self.uvc_last_frame_time = time.perf_counter()
        self.uvc_reconnect_requested = False
        camera_name = getattr(self.uvc_worker, 'camera_name', f'Camera {camera_index}')
        self.uvc_status.setText(f'{camera_name} (Camera {camera_index}) opened: {actual_width}x{actual_height}, reported {actual_fps:.1f} FPS; waiting for frame...')
        self.log.append(f'{camera_name} (Camera {camera_index}) opened: {actual_width}x{actual_height}, requested {DEFAULT_UVC_FPS} FPS, reported {actual_fps:.1f} FPS')

    def handle_uvc_worker_status(self, message):
        if self.sender() is not self.uvc_worker:
            return
        self.uvc_status.setText(message)

    def handle_uvc_stream_interrupted(self, message):
        if self.sender() is not self.uvc_worker:
            return
        self.log.append(f'UVC Algorithm Compare automatically stopped: {message}')
        if self.stop_uvc_camera(log_message=False):
            self.uvc_status.setText('Camera stopped — stream interrupted')

    def handle_uvc_worker_finished(self):
        if self.sender() is not self.uvc_worker:
            return
        self.stop_uvc_recording(reason='Camera capture thread finished')
        restart = self.uvc_restart_when_stopped
        close_pending = self.uvc_close_pending
        self.uvc_restart_when_stopped = False
        self.uvc_worker = None
        self.stop_uvc_processing_worker()
        self.uvc_watchdog_timer.stop()
        self.uvc_start_btn.setEnabled(True)
        self.uvc_stop_btn.setEnabled(False)
        self.uvc_capture_btn.setEnabled(False)
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn.setEnabled(False)
        self.uvc_actual_fps = 0.0
        self.uvc_status.setText('Camera stopped')
        if close_pending:
            self.uvc_close_pending = False
            QTimer.singleShot(0, self.close)
        elif restart and self.current_function_index() == len(self.FUNCTIONS) - 1:
            QTimer.singleShot(0, self.start_uvc_camera)

    def force_close_after_uvc_timeout(self):
        if not self.uvc_close_pending:
            return
        worker = self.uvc_worker
        if worker is not None and worker.isRunning():
            worker.terminate()
            if not worker.wait(500):
                QTimer.singleShot(500, self.force_close_after_uvc_timeout)
                return
        self.uvc_worker = None
        self.stop_uvc_processing_worker(wait_ms=500)
        self.uvc_close_pending = False
        QTimer.singleShot(0, self.close)

    def force_close_after_dual_uvc_timeout(self):
        if not self.dual_uvc_close_pending:
            return
        self.dual_uvc_display_timer.stop()
        still_running = False
        for worker in (self.dual_uvc_worker_a, self.dual_uvc_worker_b):
            if worker is not None and worker.isRunning():
                worker.terminate()
                if not worker.wait(500):
                    still_running = True
        if still_running:
            QTimer.singleShot(500, self.force_close_after_dual_uvc_timeout)
            return
        self.dual_uvc_worker_a = None
        self.dual_uvc_worker_b = None
        self.stop_dual_uvc_recording(reason='Application closing', wait_ms=500)
        self.dual_uvc_close_pending = False
        QTimer.singleShot(0, self.close)

    def check_uvc_stream(self):
        worker = self.uvc_worker
        if worker is None or not worker.isRunning():
            return
        elapsed = time.perf_counter() - self.uvc_last_frame_time
        if elapsed < UVC_FRAME_WAIT_SECONDS:
            return
        if not self.uvc_waiting_for_frame:
            self.uvc_waiting_for_frame = True
            self.log.append('UVC frame paused; keeping the last image (possible FFC).')
        self.uvc_status.setText(f'Waiting for UVC frame {elapsed:.1f}s (possible FFC); last image retained.')
        if elapsed >= UVC_RECONNECT_SECONDS and (not self.uvc_reconnect_requested):
            self.uvc_reconnect_requested = True
            camera_info = self.selected_camera_info(self.uvc_camera_combo)
            self.log.append(f"No UVC frame for {elapsed:.1f}s; requesting automatic reconnect for {camera_info['name']} (Camera {camera_info['index']}).")
            worker.force_reconnect()

    def handle_uvc_frame_seen(self, frame_time):
        if self.sender() is not self.uvc_worker:
            return
        self.uvc_last_frame_time = frame_time
        self.uvc_reconnect_requested = False
        if self.uvc_waiting_for_frame:
            self.uvc_waiting_for_frame = False
            self.log.append('UVC stream resumed.')

    def handle_uvc_frame(self, frame):
        if self.sender() is not self.uvc_worker:
            return
        processing_worker = self.uvc_processing_worker
        if processing_worker is None or not processing_worker.isRunning():
            return
        if self.app_settings['camera_mirror']['camera_a']:
            frame = np.ascontiguousarray(frame[:, ::-1])
        try:
            config = self.build_uvc_processing_config()
        except Exception as exc:
            self.handle_uvc_processing_error(str(exc))
            return
        processing_worker.submit(frame, config)

    def handle_uvc_processed_frame(self, payload):
        if self.sender() is not self.uvc_processing_worker:
            return
        if payload['revision'] != self.uvc_processing_revision:
            return
        self.uvc_last_processing_error = None
        self.uvc_original_source = payload['original_source']
        self.uvc_original_frame = payload['original_display']
        self.uvc_pipeline_input = payload['pipeline_input']
        self.uvc_pipeline_input_max = payload['pipeline_input_max']
        self.uvc_processed_frame = payload['final_image']
        self.uvc_pipeline_records = payload['records']
        self.uvc_pipeline_total_ms = payload['total_ms']
        now = time.perf_counter()
        self.current_original = self.uvc_original_frame
        self.preview_original.set_array(self.uvc_original_frame)
        self.update_uvc_stage_preview()
        self.update_uvc_video_window()
        self.uvc_capture_btn.setEnabled(True)
        recording_worker = self.uvc_recording_worker
        if recording_worker is not None and recording_worker.isRunning():
            recording_worker.submit(self.uvc_original_frame, self.uvc_processed_frame, now)
            self.uvc_record_start_btn.setEnabled(False)
            self.uvc_record_stop_btn.setEnabled(True)
        elif recording_worker is None:
            self.uvc_record_start_btn.setEnabled(True)
            self.uvc_record_stop_btn.setEnabled(False)
        else:
            self.uvc_record_start_btn.setEnabled(False)
            self.uvc_record_stop_btn.setEnabled(False)
        self.uvc_frame_count += 1
        elapsed = now - self.uvc_last_time
        if elapsed >= 0.5:
            self.uvc_fps_display = self.uvc_frame_count / elapsed
            self.uvc_frame_count = 0
            self.uvc_last_time = now
        requested_fps = DEFAULT_UVC_FPS
        budget_ms = 1000.0 / requested_fps
        budget_text = 'OVER BUDGET' if self.uvc_pipeline_total_ms > budget_ms else 'within budget'
        detail_parts = list(payload['preprocessing_details'])
        detail_parts.extend((f"S{record['pipeline_index']} {record['name']} {record['elapsed_ms']:.1f}ms" for record in self.uvc_pipeline_records))
        detail_text = ' | '.join(detail_parts)
        recording_text = ''
        if recording_worker is not None and recording_worker.isRunning():
            recording_text = f'REC {max(0.0, now - self.uvc_recording_started_perf):.1f}s | '
        selected_capture_mode = getattr(self.uvc_worker, 'selected_capture_mode', None)
        selected_capture_name = UvcCaptureThread._capture_mode_name(selected_capture_mode or DEFAULT_UVC_CAPTURE_MODE)
        camera_info = self.selected_camera_info(self.uvc_camera_combo)
        self.uvc_status.setText(recording_text + f"{camera_info['name']} (Camera {camera_info['index']}) | Capture {selected_capture_name} | {self.uvc_original_frame.shape[1]}x{self.uvc_original_frame.shape[0]} | Output FPS {self.uvc_fps_display:.1f} | Pipeline {self.uvc_pipeline_total_ms:.1f} ms / {budget_ms:.1f} ms ({budget_text})" + (f' | {detail_text}' if detail_text else ''))

    def handle_uvc_processing_error(self, message):
        if self.uvc_recording_worker is not None:
            self.stop_uvc_recording(reason='UVC processing error')
        if message != self.uvc_last_processing_error:
            self.log.append(f'UVC frame processing error: {message}')
        self.uvc_last_processing_error = message
        self.uvc_processed_frame = None
        self.uvc_pipeline_records = []
        self.current_result = None
        self.preview_result.set_array(None)
        self.uvc_capture_btn.setEnabled(False)
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn.setEnabled(False)
        self.uvc_status.setText(f'Frame processing error: {message}')

    def handle_uvc_processing_failed(self, payload):
        if self.sender() is not self.uvc_processing_worker:
            return
        if payload.get('revision') != self.uvc_processing_revision:
            return
        self.handle_uvc_processing_error(payload['message'])

    def capture_uvc_pair(self):
        if self.uvc_original_frame is None or self.uvc_processed_frame is None:
            QMessageBox.information(self, 'No UVC Frame', 'Start the camera and wait for a frame first.')
            return
        out = self.make_output('uvc_algorithm_compare')
        original_path = out / 'original.png'
        pipeline_input_path = out / 'pipeline_input.png'
        processed_path = out / 'final_pipeline.png'
        comparison_path = out / 'comparison_pipeline.png'
        overview_path = out / '00_pipeline_overview.png'
        save_png(original_path, self.uvc_original_frame)
        save_png(pipeline_input_path, self.uvc_pipeline_input)
        step_output_files = {}
        for record in self.uvc_pipeline_records:
            step_path = out / f"step_{record['pipeline_index']:02d}_{record['algorithm']}.png"
            save_png(step_path, record['image'])
            step_output_files[record['pipeline_index']] = step_path.name
        save_png(processed_path, self.uvc_processed_frame)
        save_uvc_comparison_png(comparison_path, self.uvc_original_frame, self.uvc_processed_frame, 'Final Pipeline Output')
        input_label = 'Native DN' if self.uvc_pipeline_input_max > 255 else 'UVC Gray'
        overview_items = [('00 Original UVC', self.uvc_original_frame, 255), (f'01 Pipeline Input - {input_label}', self.uvc_pipeline_input, self.uvc_pipeline_input_max)]
        for overview_index, record in enumerate(self.uvc_pipeline_records, start=2):
            overview_items.append((f"{overview_index:02d} Step {record['pipeline_index']:02d} {record['name']}", record['image'], 255))
        save_pipeline_overview_png(overview_path, overview_items)
        manifest = {'format_version': 1, 'mode': 'uvc_algorithm_compare', 'capture_mode': DEFAULT_UVC_CAPTURE_MODE, 'selected_capture_mode': getattr(self.uvc_worker, 'selected_capture_mode', None), 'camera_index': self.selected_camera_info(self.uvc_camera_combo)['index'], 'camera_name': self.selected_camera_info(self.uvc_camera_combo)['name'], 'camera_device_id': self.selected_camera_info(self.uvc_camera_combo)['device_id'], 'requested_fps': DEFAULT_UVC_FPS, 'resolution': {'width': int(self.uvc_original_frame.shape[1]), 'height': int(self.uvc_original_frame.shape[0])}, 'selected_thermal_resolution': {'width': self.width, 'height': self.height}, 'bit_depth': self.bit_depth, 'pipeline': self.uvc_pipeline_editor.configuration(), 'execution': pipeline_execution_json(self.uvc_pipeline_records), 'total_ms': round(self.uvc_pipeline_total_ms, 3), 'original_output': original_path.name, 'pipeline_input_output': pipeline_input_path.name, 'final_output': processed_path.name, 'comparison_output': comparison_path.name, 'overview_output': overview_path.name}
        for execution in manifest['execution']:
            output_file = step_output_files.get(execution['pipeline_index'])
            if output_file:
                execution['output_file'] = output_file
        with open(out / 'pipeline.json', 'w', encoding='utf-8') as destination:
            manifest['weighted_mapping_settings'] = dict(self.app_settings['weighted_mapping'])
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / 'pipeline_summary.txt', 'w', encoding='utf-8') as summary:
            summary.write(f"Camera: {manifest['camera_name']} (index {manifest['camera_index']})\n")
            summary.write(f"Camera device ID: {manifest['camera_device_id']}\n")
            summary.write(f"Capture mode: {manifest['capture_mode']}\n")
            summary.write(f"Selected capture mode: {manifest['selected_capture_mode']}\n")
            summary.write(f"Actual resolution: {manifest['resolution']['width']}x{manifest['resolution']['height']}\n")
            summary.write(f'Overview: {overview_path.name}\n')
            summary.write(f'Pipeline: {self.uvc_pipeline_editor.summary()}\n')
            for execution in manifest['execution']:
                summary.write(f"Step {execution['pipeline_index']}: {execution['name']} - {execution['elapsed_ms']:.3f} ms - {' | '.join(execution['details'])}\n")
            summary.write(f'Total: {self.uvc_pipeline_total_ms:.3f} ms\n')
        self.log.append(f'Saved: {original_path}')
        self.log.append(f'Saved: {processed_path}')
        self.log.append(f'Saved comparison: {comparison_path}')
        self.log.append(f'Saved overview: {overview_path}')
        self.log.append(f"Pipeline settings: {out / 'pipeline.json'}")
