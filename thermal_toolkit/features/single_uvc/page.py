"""features / single_uvc / page."""
from datetime import datetime
import json
import numpy as np
import time
from thermal_toolkit.algorithms.mapping import raw_to_display_u8, uvc_to_gray
from thermal_toolkit.analysis.statistics import gray_image_analysis
from thermal_toolkit.core.constants import DEFAULT_UVC_CAPTURE_MODE, DEFAULT_UVC_FPS, UVC_RECONNECT_SECONDS, UVC_VIEWER_ANALYSIS_INTERVAL_SECONDS
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import QGridLayout, QLabel, QMessageBox, QPushButton, QSizePolicy, QTimer, QVBoxLayout, QWidget, Qt
from thermal_toolkit.io.export import save_png, save_rgb_png
from thermal_toolkit.reports.histogram import render_gray_histogram_rgb, save_histogram_excel
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox
from thermal_toolkit.ui.widgets.preview import PreviewLabel, UvcVideoWindow
from thermal_toolkit.uvc.capture import UvcCaptureThread

class SingleUvcPageMixin:
    """features / single_uvc / page behavior composed by the main window."""

    def build_single_uvc_page(self):
        p, f = self._form_page()
        self.viewer_uvc_note = QLabel('Clean UVC live view using Auto Detect. Double-click the image or Histogram to open a resizable analysis window.')
        self.viewer_uvc_note.setWordWrap(True)
        self.viewer_uvc_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.viewer_uvc_note.setMaximumHeight(72)
        self.viewer_uvc_note.setStyleSheet('color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; border-radius:5px; padding:8px;')
        self.viewer_uvc_camera_combo = FocusWheelComboBox()
        self.viewer_uvc_start_btn = QPushButton('Start Camera')
        self.viewer_uvc_start_btn.setObjectName('startButton')
        self.viewer_uvc_start_btn.clicked.connect(self.start_uvc_viewer)
        self.viewer_uvc_stop_btn = QPushButton('Stop Camera')
        self.viewer_uvc_stop_btn.setObjectName('stopButton')
        self.viewer_uvc_stop_btn.clicked.connect(self.stop_uvc_viewer)
        self.viewer_uvc_stop_btn.setEnabled(False)
        self.viewer_uvc_capture_btn = QPushButton('Capture')
        self.viewer_uvc_capture_btn.setObjectName('captureButton')
        self.viewer_uvc_capture_btn.clicked.connect(self.capture_uvc_viewer)
        self.viewer_uvc_capture_btn.setEnabled(False)
        self.viewer_uvc_record_start_btn = QPushButton('Start Recording')
        self.viewer_uvc_record_start_btn.setObjectName('startButton')
        self.viewer_uvc_record_start_btn.clicked.connect(self.start_uvc_viewer_recording)
        self.viewer_uvc_record_start_btn.setEnabled(False)
        self.viewer_uvc_record_stop_btn = QPushButton('Stop Recording')
        self.viewer_uvc_record_stop_btn.setObjectName('stopButton')
        self.viewer_uvc_record_stop_btn.clicked.connect(self.stop_uvc_viewer_recording)
        self.viewer_uvc_record_stop_btn.setEnabled(False)
        self.viewer_uvc_refresh_btn = QPushButton('Refresh Cameras')
        self.viewer_uvc_refresh_btn.clicked.connect(self.refresh_camera_devices)
        viewer_controls = QWidget()
        viewer_controls_layout = QGridLayout(viewer_controls)
        viewer_controls_layout.setContentsMargins(0, 0, 0, 0)
        viewer_controls_layout.addWidget(self.viewer_uvc_start_btn, 0, 0)
        viewer_controls_layout.addWidget(self.viewer_uvc_stop_btn, 0, 1)
        viewer_controls_layout.addWidget(self.viewer_uvc_capture_btn, 1, 0, 1, 2)
        viewer_controls_layout.addWidget(self.viewer_uvc_record_start_btn, 2, 0)
        viewer_controls_layout.addWidget(self.viewer_uvc_record_stop_btn, 2, 1)
        viewer_controls_layout.addWidget(self.viewer_uvc_refresh_btn, 3, 0, 1, 2)
        self.viewer_uvc_status = QLabel('Camera stopped')
        self.viewer_uvc_status.setWordWrap(True)
        self.viewer_uvc_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow(self.viewer_uvc_note)
        f.addRow('CAMERA_INDEX', self.viewer_uvc_camera_combo)
        f.addRow(viewer_controls)
        f.addRow('Status', self.viewer_uvc_status)
        self.viewer_uvc_stats_group = CollapsibleGroupBox('Live Gray Statistics (8-bit)')
        self.viewer_uvc_stats_group.setObjectName('viewerStatsCard')
        viewer_uvc_stats_layout = QGridLayout(self.viewer_uvc_stats_group)
        viewer_uvc_stats_layout.setHorizontalSpacing(10)
        viewer_uvc_stats_layout.setVerticalSpacing(7)
        self.viewer_uvc_stat_values = {}
        viewer_uvc_stat_fields = [('Pixel Count', 'count'), ('Gray Levels', 'unique_levels'), ('Minimum', 'min'), ('Maximum', 'max'), ('Mean', 'mean'), ('Median / P50', 'median'), ('Std Dev', 'std'), ('Entropy', 'entropy_bits'), ('P1', 'p1'), ('P5', 'p5'), ('P95', 'p95'), ('P99', 'p99'), ('P99 - P1', 'p99_p1'), ('Zero Clip', 'zero_clip'), ('255 Clip', 'full_clip'), ('Y <= 16', 'limited_low'), ('Y >= 235', 'limited_high')]
        for index, (label_text, key) in enumerate(viewer_uvc_stat_fields):
            row = index // 2
            column = index % 2 * 2
            name_label = QLabel(label_text)
            value_label = QLabel('--')
            value_label.setProperty('viewerStatValue', True)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setMinimumWidth(90)
            viewer_uvc_stats_layout.addWidget(name_label, row, column)
            viewer_uvc_stats_layout.addWidget(value_label, row, column + 1)
            self.viewer_uvc_stat_values[key] = value_label
        viewer_uvc_stats_note = QLabel('Updated twice per second from every displayed pixel. Clipping shows count and frame percentage; entropy is measured in bits.')
        viewer_uvc_stats_note.setWordWrap(True)
        viewer_uvc_stats_note.setStyleSheet('color:#aeb8c2; font-weight:400; padding-top:4px;')
        viewer_uvc_stats_layout.addWidget(viewer_uvc_stats_note, (len(viewer_uvc_stat_fields) + 1) // 2, 0, 1, 4)
        f.addRow(self.viewer_uvc_stats_group)
        self.viewer_uvc_histogram_group = CollapsibleGroupBox('Live Gray Histogram + CDF')
        self.viewer_uvc_histogram_group.setObjectName('viewerStatsCard')
        viewer_uvc_histogram_layout = QVBoxLayout(self.viewer_uvc_histogram_group)
        self.viewer_uvc_histogram_preview = PreviewLabel('Live Gray Histogram')
        self.viewer_uvc_histogram_preview.setMinimumSize(320, 180)
        self.viewer_uvc_histogram_preview.setToolTip('Double-click to open a resizable and maximizable histogram window')
        self.viewer_uvc_histogram_preview.double_clicked.connect(self.open_uvc_histogram_window)
        viewer_uvc_histogram_layout.addWidget(self.viewer_uvc_histogram_preview)
        histogram_note = QLabel('Blue bars: 256-bin gray histogram. Optional yellow line: cumulative distribution (CDF), controlled by Settings > Show CDF line. Markers show P1, P50, and P99.')
        histogram_note.setWordWrap(True)
        histogram_note.setStyleSheet('color:#aeb8c2; font-weight:400; padding-top:3px;')
        viewer_uvc_histogram_layout.addWidget(histogram_note)
        f.addRow(self.viewer_uvc_histogram_group)
        self.stack.addWidget(p)
        self.viewer_uvc_camera_combo.currentIndexChanged.connect(self.switch_uvc_viewer_setting)
        self.resolution_combo.currentIndexChanged.connect(self.switch_uvc_viewer_setting)

    def clear_uvc_viewer_analysis(self, reset_timeline=True):
        self.viewer_uvc_stats = None
        self.viewer_uvc_histogram = None
        self.viewer_uvc_histogram_rgb = None
        self.viewer_uvc_last_analysis_time = 0.0
        if hasattr(self, 'viewer_uvc_stat_values'):
            for value_label in self.viewer_uvc_stat_values.values():
                value_label.setText('--')
        if hasattr(self, 'viewer_uvc_histogram_preview'):
            self.viewer_uvc_histogram_preview.set_array(None)

    def update_uvc_viewer_analysis(self, image, frame_time=None):
        stats, histogram = gray_image_analysis(image)
        self.viewer_uvc_stats = stats
        self.viewer_uvc_histogram = histogram
        self.viewer_uvc_last_analysis_time = time.perf_counter() if frame_time is None else float(frame_time)
        values = self.viewer_uvc_stat_values
        values['count'].setText(f"{stats['count']:,}")
        values['unique_levels'].setText(str(stats['unique_levels']))
        values['min'].setText(str(stats['min']))
        values['max'].setText(str(stats['max']))
        for key in ('mean', 'median', 'std', 'p1', 'p5', 'p95', 'p99'):
            values[key].setText(f'{stats[key]:.3f}')
        values['p99_p1'].setText(f"{stats['p99_p1']:.3f}")
        values['entropy_bits'].setText(f"{stats['entropy_bits']:.3f} bits")
        values['zero_clip'].setText(f"{stats['zero_count']:,} ({stats['zero_percent']:.3f}%)")
        values['full_clip'].setText(f"{stats['full_count']:,} ({stats['full_percent']:.3f}%)")
        values['limited_low'].setText(f"{stats['limited_low_count']:,} ({stats['limited_low_percent']:.3f}%)")
        values['limited_high'].setText(f"{stats['limited_high_count']:,} ({stats['limited_high_percent']:.3f}%)")
        self.refresh_uvc_histogram_cdf()

    def refresh_uvc_histogram_cdf(self):
        show_cdf = self.app_settings['timeline'].get('raw_histogram_show_cdf', False)
        self.viewer_uvc_histogram_group.setTitle('Live Gray Histogram' + (' + CDF' if show_cdf else ''))
        if self.viewer_uvc_histogram_window is not None:
            self.viewer_uvc_histogram_window.setWindowTitle(self.uvc_histogram_window_title())
        if self.viewer_uvc_stats is None or self.viewer_uvc_histogram is None:
            return
        self.viewer_uvc_histogram_rgb = render_gray_histogram_rgb(
            self.viewer_uvc_histogram, self.viewer_uvc_stats, show_cdf=show_cdf)
        self.viewer_uvc_histogram_preview.set_array(self.viewer_uvc_histogram_rgb)
        self.update_uvc_histogram_window()

    def uvc_histogram_window_title(self):
        show_cdf = self.app_settings['timeline'].get('raw_histogram_show_cdf', False)
        return 'Single UVC Statistic - Gray Histogram' + (' and CDF' if show_cdf else '')

    def open_uvc_histogram_window(self):
        if self.current_function_index() != self.FUNCTIONS.index('Single UVC Statistic'):
            return
        if self.viewer_uvc_histogram_rgb is None:
            QMessageBox.information(self, 'No Histogram', 'Start Single UVC Statistic and wait for gray statistics first.')
            return
        if self.viewer_uvc_histogram_window is None:
            self.viewer_uvc_histogram_window = UvcVideoWindow(self)
        self.viewer_uvc_histogram_window.setWindowTitle(self.uvc_histogram_window_title())
        self.viewer_uvc_histogram_window.set_bgr_frame(cv2.cvtColor(self.viewer_uvc_histogram_rgb, cv2.COLOR_RGB2BGR))
        self.viewer_uvc_histogram_window.show()
        self.viewer_uvc_histogram_window.raise_()
        self.viewer_uvc_histogram_window.activateWindow()

    def update_uvc_histogram_window(self):
        window = self.viewer_uvc_histogram_window
        if window is None or not window.isVisible() or self.viewer_uvc_histogram_rgb is None:
            return
        window.set_bgr_frame(cv2.cvtColor(self.viewer_uvc_histogram_rgb, cv2.COLOR_RGB2BGR))

    def close_uvc_histogram_window(self):
        if self.viewer_uvc_histogram_window is None:
            return
        self.viewer_uvc_histogram_window.hide()
        self.viewer_uvc_histogram_window.clear_frame()

    def uvc_viewer_camera_manifest(self):
        camera_info = self.selected_camera_info(self.viewer_uvc_camera_combo)
        resolution = None
        if self.viewer_uvc_frame is not None:
            resolution = {'width': int(self.viewer_uvc_frame.shape[1]), 'height': int(self.viewer_uvc_frame.shape[0])}
        return {'capture_mode': DEFAULT_UVC_CAPTURE_MODE, 'selected_capture_mode': getattr(self.viewer_uvc_worker, 'selected_capture_mode', None), 'camera_index': camera_info['index'], 'camera_name': camera_info['name'], 'camera_device_id': camera_info['device_id'], 'requested_fps': int(DEFAULT_UVC_FPS), 'reported_camera_fps': round(self.viewer_uvc_actual_fps, 3), 'resolution': resolution}

    @staticmethod
    def write_uvc_viewer_analysis_summary(summary, stats):
        summary.write('Gray statistics (8-bit):\n')
        summary.write(f"  Pixels / gray levels: {stats['count']:,} / {stats['unique_levels']}\n")
        summary.write(f"  Min / max: {stats['min']} / {stats['max']}\n")
        summary.write(f"  Mean / median / std: {stats['mean']:.3f} / {stats['median']:.3f} / {stats['std']:.3f}\n")
        summary.write(f"  P1 / P5 / P95 / P99: {stats['p1']:.3f} / {stats['p5']:.3f} / {stats['p95']:.3f} / {stats['p99']:.3f}\n")
        summary.write(f"  P99-P1: {stats['p99_p1']:.3f}\n")
        summary.write(f"  Entropy: {stats['entropy_bits']:.3f} bits\n")
        summary.write(f"  Zero clip: {stats['zero_count']:,} ({stats['zero_percent']:.3f}%)\n")
        summary.write(f"  255 clip: {stats['full_count']:,} ({stats['full_percent']:.3f}%)\n")
        summary.write(f"  Y<=16: {stats['limited_low_count']:,} ({stats['limited_low_percent']:.3f}%)\n")
        summary.write(f"  Y>=235: {stats['limited_high_count']:,} ({stats['limited_high_percent']:.3f}%)\n")

    def update_uvc_viewer_button_states(self):
        running = self.viewer_uvc_worker is not None
        ready = running and self.viewer_uvc_frame is not None
        recording = self.viewer_uvc_recording_worker is not None
        self.viewer_uvc_camera_combo.setEnabled(not recording)
        self.viewer_uvc_start_btn.setEnabled(not running)
        self.viewer_uvc_stop_btn.setEnabled(running)
        self.viewer_uvc_capture_btn.setEnabled(ready)
        self.viewer_uvc_record_start_btn.setEnabled(ready and (not recording))
        self.viewer_uvc_record_stop_btn.setEnabled(recording)
        if self.current_function_index() == self.FUNCTIONS.index('Single UVC Statistic'):
            self.run_btn.setEnabled(ready)

    def capture_uvc_viewer(self, _checked=False):
        if self.viewer_uvc_worker is None or self.viewer_uvc_frame is None:
            QMessageBox.information(self, 'No UVC Frame', 'Start Single UVC Statistic and wait for an image first.')
            return
        self.update_uvc_viewer_analysis(self.viewer_uvc_frame)
        out = self.make_output('uvc_viewer_capture')
        image_path = out / 'uvc_viewer.png'
        histogram_path = out / 'gray_histogram.png'
        excel_path = out / 'gray_analysis.xlsx'
        save_png(image_path, self.viewer_uvc_frame)
        save_rgb_png(histogram_path, self.viewer_uvc_histogram_rgb)
        manifest = {'format_version': 1, 'mode': 'uvc_viewer_capture', 'captured_at': datetime.now().astimezone().isoformat(timespec='seconds'), **self.uvc_viewer_camera_manifest(), 'gray_analysis': dict(self.viewer_uvc_stats), 'image_output': image_path.name, 'histogram_output': histogram_path.name, 'excel_output': excel_path.name}
        with open(out / 'capture.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / 'capture_summary.txt', 'w', encoding='utf-8') as summary:
            summary.write(f"Captured: {manifest['captured_at']}\n")
            summary.write(f"Camera: {manifest['camera_name']} (index {manifest['camera_index']})\n")
            summary.write(f"Capture mode: {manifest['capture_mode']} -> {manifest['selected_capture_mode']}\n")
            summary.write(f"Resolution: {manifest['resolution']['width']}x{manifest['resolution']['height']}\n")
            self.write_uvc_viewer_analysis_summary(summary, manifest['gray_analysis'])
            summary.write(f'Image: {image_path.name}\n')
            summary.write(f'Histogram: {histogram_path.name}\n')
            summary.write(f'Excel analysis: {excel_path.name}\n')
        save_histogram_excel(excel_path, 'Single UVC Statistic Capture Analysis', {key: value for key, value in manifest.items() if key != 'gray_analysis'}, [{'label': 'Capture', 'stats': self.viewer_uvc_stats, 'histogram': self.viewer_uvc_histogram}], value_max=255, value_name='Gray')
        self.log.append(f'Saved Single UVC Statistic capture: {image_path}')
        self.log.append(f'Saved Single UVC Statistic histogram: {histogram_path}')
        self.log.append(f'Saved Single UVC Statistic Excel analysis: {excel_path}')

    def start_uvc_viewer(self, _checked=False):
        if cv2 is None:
            message = 'OpenCV not installed. pip install opencv-python'
            self.log.append(f'\nERROR: {message}')
            QMessageBox.critical(self, 'Camera Error', message)
            return
        if not self.stop_uvc_viewer(log_message=False):
            self.viewer_uvc_restart_when_stopped = True
            self.log.append('Previous Single UVC Statistic thread is still stopping; the camera will start automatically afterward.')
            return
        camera_info = self.selected_camera_info(self.viewer_uvc_camera_combo)
        camera_index = camera_info['index']
        if camera_index is None:
            QMessageBox.warning(self, 'Camera Disconnected', 'Select a connected camera before starting Single UVC Statistic.')
            return
        requested_fps = DEFAULT_UVC_FPS
        worker = UvcCaptureThread(camera_index, self.width, self.height, requested_fps, UVC_RECONNECT_SECONDS, self, frame_consumer=self.viewer_uvc_buffer.submit, capture_mode=DEFAULT_UVC_CAPTURE_MODE, camera_device_id=camera_info['device_id'], camera_name=camera_info['name'])
        worker.camera_opened.connect(self.handle_uvc_viewer_camera_opened)
        worker.status_changed.connect(self.handle_uvc_viewer_status)
        worker.stream_interrupted.connect(self.handle_uvc_viewer_stream_interrupted)
        worker.finished.connect(self.handle_uvc_viewer_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self.viewer_uvc_worker = worker
        self.viewer_uvc_frame = None
        self.clear_uvc_viewer_analysis()
        self.viewer_uvc_actual_fps = 0.0
        self.viewer_uvc_output_fps = 0.0
        self.viewer_uvc_frame_count = 0
        self.viewer_uvc_fps_started = time.perf_counter()
        self.viewer_uvc_buffer.clear()
        self.viewer_uvc_buffer_sequence = 0
        self.preview_result.set_array(None)
        self.current_result = None
        self.update_uvc_viewer_button_states()
        self.viewer_uvc_status.setText(f"Opening {camera_info['name']} (Camera {camera_index}, {DEFAULT_UVC_CAPTURE_MODE})...")
        self.viewer_uvc_display_timer.start()
        self.log.append(f"Starting Single UVC Statistic: {camera_info['name']} (Camera {camera_index}, {requested_fps} FPS, {DEFAULT_UVC_CAPTURE_MODE})")
        worker.start()

    def stop_uvc_viewer(self, _checked=False, log_message=True, wait_ms=2000):
        self.viewer_uvc_restart_when_stopped = False
        self.close_uvc_video_window()
        self.close_uvc_histogram_window()
        self.viewer_uvc_display_timer.stop()
        if not self.stop_uvc_viewer_recording(reason='Single UVC Statistic camera stopped', log_message=log_message, wait_ms=wait_ms):
            return False
        worker = self.viewer_uvc_worker
        was_running = worker is not None
        if worker is not None:
            worker.stop()
            if not worker.wait(wait_ms):
                self.viewer_uvc_start_btn.setEnabled(False)
                self.viewer_uvc_stop_btn.setEnabled(False)
                self.viewer_uvc_status.setText('Single UVC Statistic capture thread is stopping...')
                return False
            if self.viewer_uvc_worker is worker:
                self.viewer_uvc_worker = None
        self.viewer_uvc_actual_fps = 0.0
        self.viewer_uvc_output_fps = 0.0
        self.update_uvc_viewer_button_states()
        self.viewer_uvc_status.setText('Camera stopped')
        if was_running and log_message:
            self.log.append('Single UVC Statistic camera stopped')
        return True

    def switch_uvc_viewer_setting(self, _combo_index=None):
        if self.viewer_uvc_worker is not None:
            self.start_uvc_viewer()

    def handle_uvc_viewer_camera_opened(self, camera_index, actual_width, actual_height, actual_fps):
        if self.sender() is not self.viewer_uvc_worker:
            return
        self.viewer_uvc_actual_fps = float(actual_fps)
        camera_name = getattr(self.viewer_uvc_worker, 'camera_name', f'Camera {camera_index}')
        self.viewer_uvc_status.setText(f'{camera_name} (Camera {camera_index}) opened: {actual_width}x{actual_height}, reported {actual_fps:.1f} FPS; waiting for frame...')

    def handle_uvc_viewer_status(self, message):
        if self.sender() is self.viewer_uvc_worker:
            self.viewer_uvc_status.setText(message)

    def handle_uvc_viewer_stream_interrupted(self, message):
        if self.sender() is not self.viewer_uvc_worker:
            return
        self.log.append(f'Single UVC Statistic automatically stopped: {message}')
        if self.stop_uvc_viewer(log_message=False):
            self.viewer_uvc_status.setText('Camera stopped — stream interrupted')

    def consume_uvc_viewer_latest_frame(self):
        frame, sequence, frame_time = self.viewer_uvc_buffer.take_latest(self.viewer_uvc_buffer_sequence)
        if frame is None:
            return
        self.viewer_uvc_buffer_sequence = sequence
        gray = uvc_to_gray(frame)
        display = gray.copy() if gray.dtype == np.uint8 else raw_to_display_u8(gray, self.raw_max)
        if self.app_settings['camera_mirror']['camera_a']:
            display = np.ascontiguousarray(display[:, ::-1])
        self.viewer_uvc_frame = display
        if self.viewer_uvc_last_analysis_time <= 0.0 or frame_time - self.viewer_uvc_last_analysis_time >= UVC_VIEWER_ANALYSIS_INTERVAL_SECONDS:
            self.update_uvc_viewer_analysis(display, frame_time)
        self.viewer_uvc_frame_count += 1
        elapsed = frame_time - self.viewer_uvc_fps_started
        if elapsed >= 0.5:
            self.viewer_uvc_output_fps = self.viewer_uvc_frame_count / elapsed
            self.viewer_uvc_frame_count = 0
            self.viewer_uvc_fps_started = frame_time
        if self.current_function_index() == self.FUNCTIONS.index('Single UVC Statistic'):
            self.current_result = display
            self.preview_result.set_array(display)
            self.update_uvc_video_window()
        recording_worker = self.viewer_uvc_recording_worker
        if recording_worker is not None and recording_worker.isRunning():
            recording_worker.submit(display, frame_time)
        self.update_uvc_viewer_button_states()
        camera_info = self.selected_camera_info(self.viewer_uvc_camera_combo)
        self.viewer_uvc_status.setText(f"{camera_info['name']} (Camera {camera_info['index']}) {display.shape[1]}x{display.shape[0]} | Output FPS {self.viewer_uvc_output_fps:.1f}")

    def handle_uvc_viewer_worker_finished(self):
        if self.sender() is not self.viewer_uvc_worker:
            return
        restart = self.viewer_uvc_restart_when_stopped
        self.viewer_uvc_restart_when_stopped = False
        self.close_uvc_video_window()
        self.close_uvc_histogram_window()
        self.stop_uvc_viewer_recording(reason='Single UVC Statistic capture thread finished')
        self.viewer_uvc_worker = None
        self.viewer_uvc_display_timer.stop()
        self.viewer_uvc_actual_fps = 0.0
        self.viewer_uvc_output_fps = 0.0
        self.update_uvc_viewer_button_states()
        self.viewer_uvc_status.setText('Camera stopped')
        if restart and self.current_function_index() == self.FUNCTIONS.index('Single UVC Statistic'):
            QTimer.singleShot(0, self.start_uvc_viewer)
