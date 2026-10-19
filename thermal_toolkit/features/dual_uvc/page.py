"""features / dual_uvc / page."""
from datetime import datetime
import json
import numpy as np
import time
from thermal_toolkit.algorithms.mapping import raw_to_display_u8, uvc_to_gray
from thermal_toolkit.core.constants import DEFAULT_UVC_CAPTURE_MODE, DEFAULT_UVC_FPS, DUAL_UVC_IMAGE_NAME_OPTIONS, UVC_RECONNECT_SECONDS
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import QFormLayout, QGridLayout, QLabel, QMessageBox, QPushButton, QTimer, QWidget
from thermal_toolkit.io.export import save_png, save_rgb_png
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox
from thermal_toolkit.uvc.capture import UvcCaptureThread
from thermal_toolkit.uvc.recording import UvcRecordingThread

class DualUvcPageMixin:
    """features / dual_uvc / page behavior composed by the main window."""

    def build_dual_uvc_page(self):
        p, f = self._form_page()
        self.dual_uvc_note = QLabel('Camera A and Camera B run in independent capture threads. Capture and recording pair the latest available frames in software; this is not hardware-trigger synchronization. Camera indexes must differ.')
        self.dual_uvc_note.setWordWrap(True)
        self.dual_uvc_note.setStyleSheet('color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; border-radius:5px; padding:8px;')
        f.addRow(self.dual_uvc_note)
        self.dual_uvc_group_a = CollapsibleGroupBox('UVC Camera A')
        self.dual_uvc_group_a.setObjectName('dualUvcCameraGroup')
        dual_form_a = QFormLayout(self.dual_uvc_group_a)
        dual_form_a.setContentsMargins(12, 34, 12, 12)
        dual_form_a.setHorizontalSpacing(10)
        dual_form_a.setVerticalSpacing(8)
        self.dual_uvc_image_name_a = FocusWheelComboBox()
        self.dual_uvc_image_name_b = FocusWheelComboBox()
        for image_name, file_tag in DUAL_UVC_IMAGE_NAME_OPTIONS:
            self.dual_uvc_image_name_a.addItem(image_name, file_tag)
            self.dual_uvc_image_name_b.addItem(image_name, file_tag)
        self.dual_uvc_image_name_a.setCurrentIndex(0)
        self.dual_uvc_image_name_b.setCurrentIndex(1)
        self.dual_uvc_camera_a = FocusWheelComboBox()
        self.dual_uvc_camera_b = FocusWheelComboBox()
        self.dual_uvc_status_a = QLabel('Camera A stopped')
        self.dual_uvc_status_a.setWordWrap(True)
        dual_form_a.addRow('Image Name', self.dual_uvc_image_name_a)
        dual_form_a.addRow('CAMERA_INDEX', self.dual_uvc_camera_a)
        dual_form_a.addRow('Status', self.dual_uvc_status_a)
        self.dual_uvc_group_b = CollapsibleGroupBox('UVC Camera B')
        self.dual_uvc_group_b.setObjectName('dualUvcCameraGroup')
        dual_form_b = QFormLayout(self.dual_uvc_group_b)
        dual_form_b.setContentsMargins(12, 34, 12, 12)
        dual_form_b.setHorizontalSpacing(10)
        dual_form_b.setVerticalSpacing(8)
        self.dual_uvc_status_b = QLabel('Camera B stopped')
        self.dual_uvc_status_b.setWordWrap(True)
        dual_form_b.addRow('Image Name', self.dual_uvc_image_name_b)
        dual_form_b.addRow('CAMERA_INDEX', self.dual_uvc_camera_b)
        dual_form_b.addRow('Status', self.dual_uvc_status_b)
        f.addRow(self.dual_uvc_group_a)
        f.addRow(self.dual_uvc_group_b)
        dual_controls = QWidget()
        dual_controls_layout = QGridLayout(dual_controls)
        dual_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.dual_uvc_start_btn = QPushButton('Start Cameras')
        self.dual_uvc_start_btn.setObjectName('startButton')
        self.dual_uvc_start_btn.clicked.connect(self.start_dual_uvc_cameras)
        self.dual_uvc_stop_btn = QPushButton('Stop Cameras')
        self.dual_uvc_stop_btn.setObjectName('stopButton')
        self.dual_uvc_stop_btn.clicked.connect(self.stop_dual_uvc_cameras)
        self.dual_uvc_stop_btn.setEnabled(False)
        self.dual_uvc_capture_btn = QPushButton('Capture Pair')
        self.dual_uvc_capture_btn.setObjectName('captureButton')
        self.dual_uvc_capture_btn.clicked.connect(self.capture_dual_uvc_pair)
        self.dual_uvc_capture_btn.setEnabled(False)
        self.dual_uvc_record_start_btn = QPushButton('Start Recording')
        self.dual_uvc_record_start_btn.setObjectName('startButton')
        self.dual_uvc_record_start_btn.clicked.connect(self.start_dual_uvc_recording)
        self.dual_uvc_record_start_btn.setEnabled(False)
        self.dual_uvc_record_stop_btn = QPushButton('Stop Recording')
        self.dual_uvc_record_stop_btn.setObjectName('stopButton')
        self.dual_uvc_record_stop_btn.clicked.connect(self.stop_dual_uvc_recording)
        self.dual_uvc_record_stop_btn.setEnabled(False)
        self.dual_uvc_refresh_btn = QPushButton('Refresh Cameras')
        self.dual_uvc_refresh_btn.clicked.connect(self.refresh_camera_devices)
        dual_controls_layout.addWidget(self.dual_uvc_start_btn, 0, 0)
        dual_controls_layout.addWidget(self.dual_uvc_stop_btn, 0, 1)
        dual_controls_layout.addWidget(self.dual_uvc_capture_btn, 1, 0, 1, 2)
        dual_controls_layout.addWidget(self.dual_uvc_record_start_btn, 2, 0)
        dual_controls_layout.addWidget(self.dual_uvc_record_stop_btn, 2, 1)
        dual_controls_layout.addWidget(self.dual_uvc_refresh_btn, 3, 0, 1, 2)
        f.insertRow(3, dual_controls)
        self.stack.addWidget(p)
        self.dual_uvc_camera_a.currentIndexChanged.connect(self.dual_uvc_setting_changed)
        self.dual_uvc_camera_b.currentIndexChanged.connect(self.dual_uvc_setting_changed)
        self.dual_uvc_image_name_a.currentIndexChanged.connect(self.dual_uvc_image_name_changed)
        self.dual_uvc_image_name_b.currentIndexChanged.connect(self.dual_uvc_image_name_changed)
        self.resolution_combo.currentIndexChanged.connect(self.dual_uvc_setting_changed)
        self.dual_uvc_image_name_changed()

    def dual_uvc_image_names(self):
        return (self.dual_uvc_image_name_a.currentText(), self.dual_uvc_image_name_b.currentText())

    def dual_uvc_image_file_tags(self):
        return (self.dual_uvc_image_name_a.currentData(), self.dual_uvc_image_name_b.currentData())

    def dual_uvc_image_name_changed(self, _combo_index=None):
        name_a, name_b = self.dual_uvc_image_names()
        self.dual_uvc_group_a.setTitle(f'UVC Camera A — {name_a}')
        self.dual_uvc_group_b.setTitle(f'UVC Camera B — {name_b}')
        if self.dual_uvc_worker_a is None:
            self.dual_uvc_status_a.setText(f'{name_a} (Camera A) stopped')
        if self.dual_uvc_worker_b is None:
            self.dual_uvc_status_b.setText(f'{name_b} (Camera B) stopped')
        dual_page_index = len(self.FUNCTIONS) - 3
        if self.current_function_index() == dual_page_index:
            self.preview_original.set_title(name_a)
            self.preview_result.set_title(name_b)
            self.preview_original_caption.setText(name_a)
            self.preview_result_caption.setText(name_b)
        self.update_uvc_video_window()

    def dual_uvc_setting_changed(self, _combo_index=None):
        if self.dual_uvc_worker_a is not None or self.dual_uvc_worker_b is not None:
            self.start_dual_uvc_cameras()

    def dual_uvc_frames_ready(self):
        return self.dual_uvc_worker_a is not None and self.dual_uvc_worker_b is not None and (self.dual_uvc_frame_a is not None) and (self.dual_uvc_frame_b is not None)

    def update_dual_uvc_button_states(self):
        running = self.dual_uvc_worker_a is not None or self.dual_uvc_worker_b is not None
        ready = self.dual_uvc_frames_ready()
        recording = self.dual_uvc_recording_worker is not None
        self.dual_uvc_image_name_a.setEnabled(not recording)
        self.dual_uvc_image_name_b.setEnabled(not recording)
        self.dual_uvc_start_btn.setEnabled(not running)
        self.dual_uvc_stop_btn.setEnabled(running)
        self.dual_uvc_capture_btn.setEnabled(ready)
        self.dual_uvc_record_start_btn.setEnabled(ready and (not recording))
        self.dual_uvc_record_stop_btn.setEnabled(recording)

    def start_dual_uvc_cameras(self, _checked=False):
        if cv2 is None:
            message = 'OpenCV not installed. pip install opencv-python'
            self.log.append(f'\nERROR: {message}')
            QMessageBox.critical(self, 'Camera Error', message)
            return
        camera_info_a = self.selected_camera_info(self.dual_uvc_camera_a)
        camera_info_b = self.selected_camera_info(self.dual_uvc_camera_b)
        if camera_info_a['index'] is None or camera_info_b['index'] is None:
            QMessageBox.warning(self, 'Camera Disconnected', 'Select two connected cameras before starting Dual UVC.')
            return
        camera_a = camera_info_a['index']
        camera_b = camera_info_b['index']
        if camera_a == camera_b:
            if self.dual_uvc_worker_a is not None or self.dual_uvc_worker_b is not None:
                self.stop_dual_uvc_cameras(log_message=False)
            QMessageBox.warning(self, 'Duplicate Camera Index', 'Camera A and Camera B must use different CAMERA_INDEX values.')
            return
        if not self.stop_dual_uvc_cameras(log_message=False):
            self.dual_uvc_restart_when_stopped = True
            self.log.append('Previous dual UVC capture threads are still stopping; both cameras will restart afterward.')
            return
        self.dual_uvc_restart_when_stopped = False
        worker_a = UvcCaptureThread(camera_a, self.width, self.height, int(DEFAULT_UVC_FPS), UVC_RECONNECT_SECONDS, self, frame_consumer=self.dual_uvc_buffer_a.submit, capture_mode=DEFAULT_UVC_CAPTURE_MODE, camera_device_id=camera_info_a['device_id'], camera_name=camera_info_a['name'])
        worker_b = UvcCaptureThread(camera_b, self.width, self.height, int(DEFAULT_UVC_FPS), UVC_RECONNECT_SECONDS, self, frame_consumer=self.dual_uvc_buffer_b.submit, capture_mode=DEFAULT_UVC_CAPTURE_MODE, camera_device_id=camera_info_b['device_id'], camera_name=camera_info_b['name'])
        worker_a.camera_opened.connect(self.handle_dual_uvc_camera_opened_a)
        worker_b.camera_opened.connect(self.handle_dual_uvc_camera_opened_b)
        worker_a.status_changed.connect(self.handle_dual_uvc_status_a)
        worker_b.status_changed.connect(self.handle_dual_uvc_status_b)
        worker_a.stream_interrupted.connect(self.handle_dual_uvc_stream_interrupted)
        worker_b.stream_interrupted.connect(self.handle_dual_uvc_stream_interrupted)
        worker_a.finished.connect(self.handle_dual_uvc_worker_finished)
        worker_b.finished.connect(self.handle_dual_uvc_worker_finished)
        worker_a.finished.connect(worker_a.deleteLater)
        worker_b.finished.connect(worker_b.deleteLater)
        self.dual_uvc_worker_a = worker_a
        self.dual_uvc_worker_b = worker_b
        self.dual_uvc_frame_a = None
        self.dual_uvc_frame_b = None
        self.dual_uvc_buffer_a.clear()
        self.dual_uvc_buffer_b.clear()
        self.dual_uvc_buffer_sequence_a = 0
        self.dual_uvc_buffer_sequence_b = 0
        self.dual_uvc_frame_time_a = 0.0
        self.dual_uvc_frame_time_b = 0.0
        self.dual_uvc_actual_fps_a = 0.0
        self.dual_uvc_actual_fps_b = 0.0
        self.dual_uvc_output_fps_a = 0.0
        self.dual_uvc_output_fps_b = 0.0
        self.dual_uvc_frame_count_a = 0
        self.dual_uvc_frame_count_b = 0
        now = time.perf_counter()
        self.dual_uvc_fps_started_a = now
        self.dual_uvc_fps_started_b = now
        name_a, name_b = self.dual_uvc_image_names()
        self.dual_uvc_status_a.setText(f'Opening {name_a} (Camera A, index {camera_a})...')
        self.dual_uvc_status_b.setText(f'Opening {name_b} (Camera B, index {camera_b})...')
        self.preview_original.set_array(None)
        self.preview_result.set_array(None)
        self.update_dual_uvc_button_states()
        self.dual_uvc_display_timer.start()
        self.log.append(f"Starting Dual UVC: A={name_a}, {camera_info_a['name']}, index {camera_a} ({DEFAULT_UVC_CAPTURE_MODE}), B={name_b}, {camera_info_b['name']}, index {camera_b} ({DEFAULT_UVC_CAPTURE_MODE})")
        worker_a.start()
        worker_b.start()

    def stop_dual_uvc_cameras(self, _checked=False, log_message=True, wait_ms=2000):
        self.dual_uvc_restart_when_stopped = False
        self.close_uvc_video_window()
        self.dual_uvc_display_timer.stop()
        self.stop_dual_uvc_recording(reason='Dual UVC cameras stopped', log_message=log_message)
        worker_a = self.dual_uvc_worker_a
        worker_b = self.dual_uvc_worker_b
        was_running = worker_a is not None or worker_b is not None
        for worker in (worker_a, worker_b):
            if worker is not None:
                worker.stop()
        all_stopped = True
        if worker_a is not None:
            if worker_a.wait(wait_ms):
                if self.dual_uvc_worker_a is worker_a:
                    self.dual_uvc_worker_a = None
            else:
                all_stopped = False
        if worker_b is not None:
            if worker_b.wait(wait_ms):
                if self.dual_uvc_worker_b is worker_b:
                    self.dual_uvc_worker_b = None
            else:
                all_stopped = False
        if not all_stopped:
            self.dual_uvc_start_btn.setEnabled(False)
            self.dual_uvc_stop_btn.setEnabled(False)
            self.dual_uvc_capture_btn.setEnabled(False)
            self.dual_uvc_record_start_btn.setEnabled(False)
            self.dual_uvc_record_stop_btn.setEnabled(False)
            self.log.append('Dual UVC capture threads are still stopping.')
            return False
        self.dual_uvc_actual_fps_a = 0.0
        self.dual_uvc_actual_fps_b = 0.0
        name_a, name_b = self.dual_uvc_image_names()
        self.dual_uvc_status_a.setText(f'{name_a} (Camera A) stopped')
        self.dual_uvc_status_b.setText(f'{name_b} (Camera B) stopped')
        self.update_dual_uvc_button_states()
        if was_running and log_message:
            self.log.append('Dual UVC cameras stopped')
        return True

    def handle_dual_uvc_camera_opened_a(self, camera_index, actual_width, actual_height, actual_fps):
        if self.sender() is not self.dual_uvc_worker_a:
            return
        self.dual_uvc_actual_fps_a = float(actual_fps)
        name_a = self.dual_uvc_image_name_a.currentText()
        camera_name = getattr(self.dual_uvc_worker_a, 'camera_name', f'Camera {camera_index}')
        self.dual_uvc_status_a.setText(f'{name_a} (Camera A): {camera_name}, index {camera_index}, {actual_width}x{actual_height}, reported {actual_fps:.1f} FPS')

    def handle_dual_uvc_camera_opened_b(self, camera_index, actual_width, actual_height, actual_fps):
        if self.sender() is not self.dual_uvc_worker_b:
            return
        self.dual_uvc_actual_fps_b = float(actual_fps)
        name_b = self.dual_uvc_image_name_b.currentText()
        camera_name = getattr(self.dual_uvc_worker_b, 'camera_name', f'Camera {camera_index}')
        self.dual_uvc_status_b.setText(f'{name_b} (Camera B): {camera_name}, index {camera_index}, {actual_width}x{actual_height}, reported {actual_fps:.1f} FPS')

    def handle_dual_uvc_status_a(self, message):
        if self.sender() is self.dual_uvc_worker_a:
            self.dual_uvc_status_a.setText(message)

    def handle_dual_uvc_status_b(self, message):
        if self.sender() is self.dual_uvc_worker_b:
            self.dual_uvc_status_b.setText(message)

    def handle_dual_uvc_stream_interrupted(self, message):
        if self.sender() not in (self.dual_uvc_worker_a, self.dual_uvc_worker_b):
            return
        self.log.append(f'Dual UVC automatically stopped: {message}')
        if self.stop_dual_uvc_cameras(log_message=False):
            name_a, name_b = self.dual_uvc_image_names()
            self.dual_uvc_status_a.setText(f'{name_a} (Camera A) stopped — stream interrupted')
            self.dual_uvc_status_b.setText(f'{name_b} (Camera B) stopped — stream interrupted')

    def consume_dual_uvc_latest_frames(self):
        updated = False
        frame_a, sequence_a, frame_time_a = self.dual_uvc_buffer_a.take_latest(self.dual_uvc_buffer_sequence_a)
        if frame_a is not None:
            self.dual_uvc_buffer_sequence_a = sequence_a
            self.handle_dual_uvc_frame('a', frame_a, frame_time_a, submit_recording=False)
            updated = True
        frame_b, sequence_b, frame_time_b = self.dual_uvc_buffer_b.take_latest(self.dual_uvc_buffer_sequence_b)
        if frame_b is not None:
            self.dual_uvc_buffer_sequence_b = sequence_b
            self.handle_dual_uvc_frame('b', frame_b, frame_time_b, submit_recording=False)
            updated = True
        if updated:
            self.submit_dual_uvc_recording_pair()

    def handle_dual_uvc_frame_a(self, frame):
        if self.sender() is not self.dual_uvc_worker_a:
            return
        self.handle_dual_uvc_frame('a', frame)

    def handle_dual_uvc_frame_b(self, frame):
        if self.sender() is not self.dual_uvc_worker_b:
            return
        self.handle_dual_uvc_frame('b', frame)

    def handle_dual_uvc_frame(self, camera_key, frame, frame_time=None, submit_recording=True):
        gray = uvc_to_gray(frame)
        display = gray.copy() if gray.dtype == np.uint8 else raw_to_display_u8(gray, self.raw_max)
        mirror_key = 'camera_a' if camera_key == 'a' else 'camera_b'
        if self.app_settings['camera_mirror'][mirror_key]:
            display = np.ascontiguousarray(display[:, ::-1])
        now = time.perf_counter() if frame_time is None else float(frame_time)
        if camera_key == 'a':
            self.dual_uvc_frame_a = display
            self.dual_uvc_frame_time_a = now
            self.dual_uvc_frame_count_a += 1
            elapsed = now - self.dual_uvc_fps_started_a
            if elapsed >= 0.5:
                self.dual_uvc_output_fps_a = self.dual_uvc_frame_count_a / elapsed
                self.dual_uvc_frame_count_a = 0
                self.dual_uvc_fps_started_a = now
            self.dual_uvc_status_a.setText(f'{self.dual_uvc_image_name_a.currentText()} (Camera A) {display.shape[1]}x{display.shape[0]} | Output FPS {self.dual_uvc_output_fps_a:.1f}')
        else:
            self.dual_uvc_frame_b = display
            self.dual_uvc_frame_time_b = now
            self.dual_uvc_frame_count_b += 1
            elapsed = now - self.dual_uvc_fps_started_b
            if elapsed >= 0.5:
                self.dual_uvc_output_fps_b = self.dual_uvc_frame_count_b / elapsed
                self.dual_uvc_frame_count_b = 0
                self.dual_uvc_fps_started_b = now
            self.dual_uvc_status_b.setText(f'{self.dual_uvc_image_name_b.currentText()} (Camera B) {display.shape[1]}x{display.shape[0]} | Output FPS {self.dual_uvc_output_fps_b:.1f}')
        dual_page_index = len(self.FUNCTIONS) - 3
        if self.current_function_index() == dual_page_index:
            if self.dual_uvc_frame_a is not None:
                self.current_original = self.dual_uvc_frame_a
                self.preview_original.set_array(self.dual_uvc_frame_a)
            if self.dual_uvc_frame_b is not None:
                self.current_result = self.dual_uvc_frame_b
                self.preview_result.set_array(self.dual_uvc_frame_b)
        recording_worker = self.dual_uvc_recording_worker
        if submit_recording and recording_worker is not None and recording_worker.isRunning() and (self.dual_uvc_frame_a is not None) and (self.dual_uvc_frame_b is not None):
            recording_worker.submit(self.dual_uvc_frame_a, self.dual_uvc_frame_b, time.perf_counter())
        self.update_dual_uvc_button_states()
        self.update_uvc_video_window()

    def submit_dual_uvc_recording_pair(self):
        recording_worker = self.dual_uvc_recording_worker
        if recording_worker is not None and recording_worker.isRunning() and (self.dual_uvc_frame_a is not None) and (self.dual_uvc_frame_b is not None):
            recording_worker.submit(self.dual_uvc_frame_a, self.dual_uvc_frame_b, time.perf_counter())

    def handle_dual_uvc_worker_finished(self):
        worker = self.sender()
        if worker is self.dual_uvc_worker_a:
            self.dual_uvc_worker_a = None
            self.dual_uvc_status_a.setText(f'{self.dual_uvc_image_name_a.currentText()} (Camera A) stopped')
        elif worker is self.dual_uvc_worker_b:
            self.dual_uvc_worker_b = None
            self.dual_uvc_status_b.setText(f'{self.dual_uvc_image_name_b.currentText()} (Camera B) stopped')
        else:
            return
        self.stop_dual_uvc_recording(reason='A Dual UVC capture thread finished')
        self.update_dual_uvc_button_states()
        if self.dual_uvc_close_pending and (self.dual_uvc_worker_a is None and self.dual_uvc_worker_b is None):
            self.dual_uvc_display_timer.stop()
            self.dual_uvc_close_pending = False
            QTimer.singleShot(0, self.close)
        elif self.dual_uvc_restart_when_stopped and (self.dual_uvc_worker_a is None and self.dual_uvc_worker_b is None):
            self.dual_uvc_display_timer.stop()
            self.dual_uvc_restart_when_stopped = False
            QTimer.singleShot(0, self.start_dual_uvc_cameras)

    def dual_uvc_labels(self):
        return self.dual_uvc_image_names()

    def save_dual_uvc_overview(self, path):
        if self.dual_uvc_frame_a is None or self.dual_uvc_frame_b is None:
            raise ValueError('Both Dual UVC frames are required for overview.')
        label_a, label_b = self.dual_uvc_labels()
        comparison_bgr = UvcRecordingThread._compose_frame(self.dual_uvc_frame_a, self.dual_uvc_frame_b, label_a, label_b)
        save_rgb_png(path, cv2.cvtColor(comparison_bgr, cv2.COLOR_BGR2RGB))

    def dual_uvc_camera_manifest(self):
        camera_info_a = self.selected_camera_info(self.dual_uvc_camera_a)
        camera_info_b = self.selected_camera_info(self.dual_uvc_camera_b)
        return {'camera_a': {'image_name': self.dual_uvc_image_name_a.currentText(), 'image_name_file_tag': self.dual_uvc_image_name_a.currentData(), 'camera_index': camera_info_a['index'], 'camera_name': camera_info_a['name'], 'camera_device_id': camera_info_a['device_id'], 'capture_mode': DEFAULT_UVC_CAPTURE_MODE, 'selected_capture_mode': getattr(self.dual_uvc_worker_a, 'selected_capture_mode', None), 'requested_fps': int(DEFAULT_UVC_FPS), 'reported_fps': round(self.dual_uvc_actual_fps_a, 3), 'resolution': {'width': int(self.dual_uvc_frame_a.shape[1]), 'height': int(self.dual_uvc_frame_a.shape[0])}}, 'camera_b': {'image_name': self.dual_uvc_image_name_b.currentText(), 'image_name_file_tag': self.dual_uvc_image_name_b.currentData(), 'camera_index': camera_info_b['index'], 'camera_name': camera_info_b['name'], 'camera_device_id': camera_info_b['device_id'], 'capture_mode': DEFAULT_UVC_CAPTURE_MODE, 'selected_capture_mode': getattr(self.dual_uvc_worker_b, 'selected_capture_mode', None), 'requested_fps': int(DEFAULT_UVC_FPS), 'reported_fps': round(self.dual_uvc_actual_fps_b, 3), 'resolution': {'width': int(self.dual_uvc_frame_b.shape[1]), 'height': int(self.dual_uvc_frame_b.shape[0])}}, 'software_pair_delta_ms': round(abs(self.dual_uvc_frame_time_a - self.dual_uvc_frame_time_b) * 1000.0, 3)}

    def capture_dual_uvc_pair(self):
        if not self.dual_uvc_frames_ready():
            QMessageBox.information(self, 'No Dual UVC Frames', 'Start both cameras and wait for Camera A and Camera B frames.')
            return
        out = self.make_output('dual_uvc_compare')
        file_tag_a, file_tag_b = self.dual_uvc_image_file_tags()
        camera_a_path = out / f'camera_{file_tag_a}.png'
        camera_b_path = out / f'camera_{file_tag_b}.png'
        overview_path = out / '00_dual_uvc_overview.png'
        save_png(camera_a_path, self.dual_uvc_frame_a)
        save_png(camera_b_path, self.dual_uvc_frame_b)
        self.save_dual_uvc_overview(overview_path)
        manifest = {'format_version': 1, 'mode': 'dual_uvc_compare', 'captured_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'pairing': 'latest-frame software pairing; not hardware synchronized', **self.dual_uvc_camera_manifest(), 'camera_a_output': camera_a_path.name, 'camera_b_output': camera_b_path.name, 'overview_output': overview_path.name}
        with open(out / 'capture.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / 'capture_summary.txt', 'w', encoding='utf-8') as summary:
            summary.write(f"Captured: {manifest['captured_at']}\n")
            summary.write(f"{manifest['camera_a']['image_name']}: {manifest['camera_a']['camera_name']}, index {manifest['camera_a']['camera_index']}, {manifest['camera_a']['resolution']['width']}x{manifest['camera_a']['resolution']['height']}, requested {manifest['camera_a']['capture_mode']}, selected {manifest['camera_a']['selected_capture_mode']}\n")
            summary.write(f"{manifest['camera_b']['image_name']}: {manifest['camera_b']['camera_name']}, index {manifest['camera_b']['camera_index']}, {manifest['camera_b']['resolution']['width']}x{manifest['camera_b']['resolution']['height']}, requested {manifest['camera_b']['capture_mode']}, selected {manifest['camera_b']['selected_capture_mode']}\n")
            summary.write(f"Software pair delta: {manifest['software_pair_delta_ms']:.3f} ms\n")
            summary.write(f'Overview: {overview_path.name}\n')
        name_a, name_b = self.dual_uvc_image_names()
        self.log.append(f'Saved Dual UVC {name_a}: {camera_a_path}')
        self.log.append(f'Saved Dual UVC {name_b}: {camera_b_path}')
        self.log.append(f'Saved Dual UVC overview: {overview_path}')
