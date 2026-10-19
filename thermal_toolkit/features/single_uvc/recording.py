"""features / single_uvc / recording."""
from datetime import datetime
import json
import time
from thermal_toolkit.core.constants import DEFAULT_UVC_FPS, UVC_RECORDING_FOURCC
from thermal_toolkit.core.qt import QMessageBox
from thermal_toolkit.io.export import save_png, save_rgb_png
from thermal_toolkit.reports.histogram import save_histogram_excel
from thermal_toolkit.uvc.fps_log import UvcFpsCsvLogger
from thermal_toolkit.uvc.recording import UvcViewerRecordingThread

class SingleUvcRecordingMixin:
    """features / single_uvc / recording behavior composed by the main window."""

    def start_uvc_viewer_recording(self, _checked=False):
        if self.viewer_uvc_recording_worker is not None:
            QMessageBox.information(self, 'Recording Active', 'Single UVC Statistic recording is already active.')
            return
        if self.viewer_uvc_worker is None or self.viewer_uvc_frame is None:
            QMessageBox.information(self, 'No UVC Frame', 'Start Single UVC Statistic and wait for an image first.')
            return
        self.update_uvc_viewer_analysis(self.viewer_uvc_frame)
        requested_fps = float(DEFAULT_UVC_FPS)
        recording_fps = self.viewer_uvc_actual_fps if 1.0 <= self.viewer_uvc_actual_fps <= 240.0 else requested_fps
        out = self.make_output('uvc_viewer_recording')
        video_path = out / 'uvc_viewer.avi'
        overview_path = out / '00_uvc_viewer_overview.png'
        histogram_path = out / '00_uvc_viewer_histogram.png'
        save_png(overview_path, self.viewer_uvc_frame)
        save_rgb_png(histogram_path, self.viewer_uvc_histogram_rgb)
        started_at = datetime.now().astimezone().isoformat(timespec='seconds')
        manifest = {'format_version': 1, 'mode': 'uvc_viewer_recording', 'started_at': started_at, **self.uvc_viewer_camera_manifest(), 'gray_analysis_at_start': dict(self.viewer_uvc_stats), 'recording_fps': round(recording_fps, 3), 'recording_layout': 'Single UVC Statistic image; no text overlay', 'recording_output': video_path.name, 'received_fps_output': 'received_fps.csv', 'overview_output': overview_path.name, 'histogram_at_start_output': histogram_path.name}
        with open(out / 'recording.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        worker = UvcViewerRecordingThread(video_path, recording_fps, self)
        worker.finished.connect(self.handle_uvc_viewer_recording_finished)
        worker.finished.connect(worker.deleteLater)
        try:
            fps_path = out / 'received_fps.csv'
            logger = UvcFpsCsvLogger(fps_path, self.viewer_uvc_worker.status_changed.emit)
        except Exception as exc:
            worker.deleteLater()
            QMessageBox.critical(self, 'FPS Logging', f'Cannot create FPS log: {exc}')
            return
        self.viewer_uvc_fps_logger = logger
        self.viewer_uvc_worker.fps_logger = logger
        self.log.append(f'Recording received FPS CSV: {fps_path}')
        self.viewer_uvc_recording_worker = worker
        self.viewer_uvc_recording_output_dir = out
        self.viewer_uvc_recording_manifest = manifest
        self.viewer_uvc_recording_start_histogram = self.viewer_uvc_histogram.copy()
        self.viewer_uvc_recording_stop_reason = None
        self.viewer_uvc_recording_started_perf = time.perf_counter()
        self.update_uvc_viewer_button_states()
        worker.start()
        worker.submit(self.viewer_uvc_frame, self.viewer_uvc_recording_started_perf)
        self.log.append(f'Single UVC Statistic recording started: {video_path} ({recording_fps:.3f} FPS, {UVC_RECORDING_FOURCC})')
        self.log.append(f'Saved recording overview: {overview_path}')
        self.log.append(f'Saved recording histogram: {histogram_path}')

    def write_uvc_viewer_recording_summary(self, worker, reason):
        out = self.viewer_uvc_recording_output_dir
        manifest = self.viewer_uvc_recording_manifest
        if out is None or manifest is None:
            return
        recording_stats = worker.result_summary or {}
        self.update_uvc_viewer_analysis(self.viewer_uvc_frame)
        final_histogram_path = out / 'uvc_viewer_histogram_final.png'
        save_rgb_png(final_histogram_path, self.viewer_uvc_histogram_rgb)
        final_histogram = self.viewer_uvc_histogram.copy()
        excel_path = out / 'recording_analysis.xlsx'
        manifest['stopped_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        manifest['stop_reason'] = reason
        manifest['gray_analysis_at_stop'] = dict(self.viewer_uvc_stats)
        manifest['histogram_at_stop_output'] = final_histogram_path.name
        manifest['recording'] = recording_stats
        manifest['excel_output'] = excel_path.name
        with open(out / 'recording.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / 'recording_summary.txt', 'w', encoding='utf-8') as summary:
            summary.write(f"Started: {manifest['started_at']}\n")
            summary.write(f"Stopped: {manifest['stopped_at']}\n")
            summary.write(f'Reason: {reason}\n')
            summary.write(f"Camera: {manifest['camera_name']} (index {manifest['camera_index']})\n")
            summary.write(f"Capture mode: {manifest['capture_mode']} -> {manifest['selected_capture_mode']}\n")
            summary.write(f"Resolution: {manifest['resolution']['width']}x{manifest['resolution']['height']}\n")
            summary.write(f"Recording FPS: {manifest['recording_fps']:.3f}\n")
            summary.write('Analysis at start:\n')
            self.write_uvc_viewer_analysis_summary(summary, manifest['gray_analysis_at_start'])
            summary.write('Analysis at stop:\n')
            self.write_uvc_viewer_analysis_summary(summary, manifest['gray_analysis_at_stop'])
            summary.write(f"Codec: {recording_stats.get('codec', 'N/A')}\n")
            summary.write(f"Duration: {recording_stats.get('duration_seconds', 0.0):.3f} s\n")
            summary.write(f"Submitted / written frames: {recording_stats.get('submitted_frames', 0)} / {recording_stats.get('written_frames', 0)}\n")
            summary.write(f"Duplicate / queue dropped / timestamp dropped: {recording_stats.get('duplicate_frames', 0)} / {recording_stats.get('queue_dropped_frames', 0)} / {recording_stats.get('timestamp_dropped_frames', 0)}\n")
            summary.write(f"File size: {recording_stats.get('file_size_bytes', 0)} bytes\n")
            summary.write(f"Video: {manifest['recording_output']}\n")
            summary.write(f"Overview: {manifest['overview_output']}\n")
            summary.write(f"Histogram at start: {manifest['histogram_at_start_output']}\n")
            summary.write(f"Histogram at stop: {manifest['histogram_at_stop_output']}\n")
            summary.write(f'Excel analysis: {excel_path.name}\n')
        start_histogram = self.viewer_uvc_recording_start_histogram
        if start_histogram is None:
            start_histogram = final_histogram
        save_histogram_excel(excel_path, 'Single UVC Statistic Recording Analysis', {key: value for key, value in manifest.items() if key not in {'gray_analysis_at_start', 'gray_analysis_at_stop'}}, [{'label': 'Start', 'stats': manifest['gray_analysis_at_start'], 'histogram': start_histogram}, {'label': 'Stop', 'stats': manifest['gray_analysis_at_stop'], 'histogram': final_histogram}], value_max=255, value_name='Gray')
        self.log.append(f'Saved Single UVC Statistic recording Excel analysis: {excel_path}')

    def close_uvc_viewer_fps_log(self):
        logger = getattr(self, 'viewer_uvc_fps_logger', None)
        if logger is not None:
            logger.close()
            self.viewer_uvc_fps_logger = None

    def finalize_uvc_viewer_recording_worker(self, worker, reason):
        if worker is not self.viewer_uvc_recording_worker:
            return
        self.close_uvc_viewer_fps_log()
        try:
            self.write_uvc_viewer_recording_summary(worker, reason)
        except Exception as exc:
            self.log.append(f'Single UVC Statistic recording summary error: {exc}')
        out = self.viewer_uvc_recording_output_dir
        if worker.error_message:
            self.log.append(f'Single UVC Statistic recording error: {worker.error_message}')
        elif out is not None:
            stats = worker.result_summary or {}
            self.log.append(f"Single UVC Statistic recording stopped: {out / 'uvc_viewer.avi'}; {stats.get('written_frames', 0)} frames, {stats.get('file_size_bytes', 0)} bytes")
        self.viewer_uvc_recording_worker = None
        self.viewer_uvc_recording_output_dir = None
        self.viewer_uvc_recording_manifest = None
        self.viewer_uvc_recording_start_histogram = None
        self.viewer_uvc_recording_stop_reason = None
        self.viewer_uvc_recording_started_perf = 0.0
        self.update_uvc_viewer_button_states()

    def stop_uvc_viewer_recording(self, _checked=False, reason=None, log_message=True, wait_ms=3000):
        self.close_uvc_viewer_fps_log()
        worker = self.viewer_uvc_recording_worker
        if worker is None:
            return True
        stop_reason = reason or 'Stopped by user'
        self.viewer_uvc_recording_stop_reason = stop_reason
        self.viewer_uvc_record_start_btn.setEnabled(False)
        self.viewer_uvc_record_stop_btn.setEnabled(False)
        worker.stop()
        if worker.isRunning() and (not worker.wait(wait_ms)):
            if log_message:
                self.log.append('Single UVC Statistic recording thread is still finishing the AVI file.')
            return False
        self.finalize_uvc_viewer_recording_worker(worker, stop_reason)
        return True

    def handle_uvc_viewer_recording_finished(self):
        worker = self.sender()
        if worker is not self.viewer_uvc_recording_worker:
            return
        reason = self.viewer_uvc_recording_stop_reason
        if reason is None:
            reason = 'Recording error' if worker.error_message else 'Recording worker finished'
        self.finalize_uvc_viewer_recording_worker(worker, reason)
