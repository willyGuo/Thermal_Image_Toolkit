"""features / uvc_algorithm / recording."""
from datetime import datetime
import json
import time
from thermal_toolkit.algorithms.pipeline import pipeline_execution_json
from thermal_toolkit.core.constants import DEFAULT_UVC_CAPTURE_MODE, DEFAULT_UVC_FPS, UVC_RECORDING_FOURCC
from thermal_toolkit.core.qt import QMessageBox
from thermal_toolkit.io.export import save_png
from thermal_toolkit.reports.pipeline import save_pipeline_overview_png
from thermal_toolkit.uvc.recording import UvcRecordingThread

class UvcAlgorithmRecordingMixin:
    """features / uvc_algorithm / recording behavior composed by the main window."""

    def start_uvc_recording(self, _checked=False):
        if self.uvc_recording_worker is not None:
            QMessageBox.information(self, 'Recording Active', 'UVC recording is already active.')
            return
        if self.uvc_worker is None or self.uvc_original_frame is None or self.uvc_processed_frame is None or (self.uvc_pipeline_input is None):
            QMessageBox.information(self, 'No UVC Frame', 'Start the camera and wait for both previews first.')
            return
        requested_fps = float(DEFAULT_UVC_FPS)
        recording_fps = self.uvc_actual_fps if 1.0 <= self.uvc_actual_fps <= 240.0 else requested_fps
        started_at = datetime.now().astimezone().isoformat(timespec='seconds')
        out = self.make_output('uvc_recording')
        video_path = out / 'comparison_pipeline.avi'
        overview_path = out / '00_pipeline_overview.png'
        manifest_path = out / 'pipeline.json'
        try:
            save_pipeline_overview_png(overview_path, self.build_uvc_overview_items())
            manifest = {'format_version': 1, 'mode': 'uvc_recording', 'started_at': started_at, 'capture_mode': DEFAULT_UVC_CAPTURE_MODE, 'selected_capture_mode': getattr(self.uvc_worker, 'selected_capture_mode', None), 'camera_index': self.selected_camera_info(self.uvc_camera_combo)['index'], 'camera_name': self.selected_camera_info(self.uvc_camera_combo)['name'], 'camera_device_id': self.selected_camera_info(self.uvc_camera_combo)['device_id'], 'requested_fps': requested_fps, 'reported_camera_fps': round(self.uvc_actual_fps, 3), 'recording_fps': round(recording_fps, 3), 'source_resolution': {'width': int(self.uvc_original_frame.shape[1]), 'height': int(self.uvc_original_frame.shape[0])}, 'recording_layout': 'Original UVC | Final Pipeline Output', 'recording_output': video_path.name, 'overview_output': overview_path.name, 'bit_depth': self.bit_depth, 'pipeline': self.uvc_pipeline_editor.configuration(), 'execution_at_start': pipeline_execution_json(self.uvc_pipeline_records)}
            with open(manifest_path, 'w', encoding='utf-8') as destination:
                json.dump(manifest, destination, ensure_ascii=False, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, 'Recording Error', str(exc))
            self.log.append(f'UVC recording setup error: {exc}')
            return
        worker = UvcRecordingThread(video_path, recording_fps, self)
        worker.finished.connect(self.handle_uvc_recording_finished)
        worker.finished.connect(worker.deleteLater)
        self.uvc_recording_worker = worker
        self.uvc_recording_output_dir = out
        self.uvc_recording_manifest = manifest
        self.uvc_recording_stop_reason = None
        self.uvc_recording_started_perf = time.perf_counter()
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn.setEnabled(True)
        worker.start()
        worker.submit(self.uvc_original_frame, self.uvc_processed_frame, self.uvc_recording_started_perf)
        self.log.append(f'UVC recording started: {video_path} ({recording_fps:.3f} FPS, {UVC_RECORDING_FOURCC})')
        self.log.append(f'Saved recording overview: {overview_path}')

    def write_uvc_recording_summary(self, worker, reason):
        out = self.uvc_recording_output_dir
        manifest = self.uvc_recording_manifest
        if out is None or manifest is None:
            return
        stats = worker.result_summary or {}
        error_message = worker.error_message
        manifest['ended_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        manifest['stop_reason'] = reason
        manifest['success'] = error_message is None and bool(stats.get('written_frames', 0))
        manifest['error'] = error_message
        manifest['recording'] = stats
        recorded_pipeline_text = ' -> '.join((stage['name'] for stage in manifest['pipeline'] if stage.get('enabled')))
        with open(out / 'pipeline.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / 'recording_summary.txt', 'w', encoding='utf-8') as summary:
            summary.write(f"Started: {manifest['started_at']}\n")
            summary.write(f"Ended: {manifest['ended_at']}\n")
            summary.write(f'Stop reason: {reason}\n')
            summary.write(f"Success: {manifest['success']}\n")
            if error_message:
                summary.write(f'Error: {error_message}\n')
            summary.write(f"Camera: {manifest['camera_name']} (index {manifest['camera_index']})\n")
            summary.write(f"Camera device ID: {manifest['camera_device_id']}\n")
            summary.write(f"Capture mode: {manifest['capture_mode']}\n")
            summary.write(f"Selected capture mode: {manifest['selected_capture_mode']}\n")
            summary.write(f"Source resolution: {manifest['source_resolution']['width']}x{manifest['source_resolution']['height']}\n")
            summary.write(f"Requested / reported / recording FPS: {manifest['requested_fps']:.3f} / {manifest['reported_camera_fps']:.3f} / {manifest['recording_fps']:.3f}\n")
            summary.write(f"Codec: {stats.get('codec', '--')}\n")
            summary.write(f"Duration: {stats.get('duration_seconds', 0.0):.3f} s\n")
            summary.write(f"Submitted / written / duplicate frames: {stats.get('submitted_frames', 0)} / {stats.get('written_frames', 0)} / {stats.get('duplicate_frames', 0)}\n")
            summary.write(f"Queue / timestamp dropped frames: {stats.get('queue_dropped_frames', 0)} / {stats.get('timestamp_dropped_frames', 0)}\n")
            summary.write(f"File size: {stats.get('file_size_bytes', 0)} bytes\n")
            summary.write(f"Overview: {manifest['overview_output']}\n")
            summary.write(f'Pipeline: {recorded_pipeline_text}\n')

    def finalize_uvc_recording_worker(self, worker, reason):
        if worker is not self.uvc_recording_worker:
            return
        try:
            self.write_uvc_recording_summary(worker, reason)
        except Exception as exc:
            self.log.append(f'UVC recording summary error: {exc}')
        out = self.uvc_recording_output_dir
        video_path = None if out is None else out / 'comparison_pipeline.avi'
        if worker.error_message:
            self.log.append(f'UVC recording error: {worker.error_message}')
            self.uvc_status.setText(f'Recording error: {worker.error_message}')
        else:
            stats = worker.result_summary or {}
            self.log.append(f"UVC recording stopped: {video_path}; written={stats.get('written_frames', 0)}, duplicates={stats.get('duplicate_frames', 0)}, dropped={stats.get('queue_dropped_frames', 0) + stats.get('timestamp_dropped_frames', 0)}")
            self.uvc_status.setText(f'Recording saved: {video_path}')
        self.uvc_recording_worker = None
        self.uvc_recording_output_dir = None
        self.uvc_recording_manifest = None
        self.uvc_recording_stop_reason = None
        self.uvc_recording_started_perf = 0.0
        frame_ready = self.uvc_worker is not None and self.uvc_original_frame is not None and (self.uvc_processed_frame is not None)
        self.uvc_record_start_btn.setEnabled(frame_ready)
        self.uvc_record_stop_btn.setEnabled(False)

    def stop_uvc_recording(self, _checked=False, reason=None, wait_ms=5000, log_message=True):
        worker = self.uvc_recording_worker
        if worker is None:
            return True
        stop_reason = reason or 'User requested'
        self.uvc_recording_stop_reason = stop_reason
        worker.stop()
        self.uvc_record_start_btn.setEnabled(False)
        self.uvc_record_stop_btn.setEnabled(False)
        if not worker.wait(wait_ms):
            self.uvc_status.setText('Finishing UVC recording...')
            if log_message:
                self.log.append('UVC recording thread is still finishing the AVI file.')
            return False
        self.finalize_uvc_recording_worker(worker, stop_reason)
        return True

    def handle_uvc_recording_finished(self):
        worker = self.sender()
        if worker is not self.uvc_recording_worker:
            return
        reason = self.uvc_recording_stop_reason
        if reason is None:
            reason = 'Recording error' if worker.error_message else 'Recording worker finished'
        self.finalize_uvc_recording_worker(worker, reason)
