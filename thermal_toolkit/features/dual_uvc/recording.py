"""features / dual_uvc / recording."""
from datetime import datetime
import json
import time
from thermal_toolkit.core.constants import DEFAULT_UVC_FPS, UVC_RECORDING_FOURCC
from thermal_toolkit.core.qt import QMessageBox
from thermal_toolkit.io.export import save_png
from thermal_toolkit.uvc.recording import UvcRecordingThread

class DualUvcRecordingMixin:
    """features / dual_uvc / recording behavior composed by the main window."""

    def start_dual_uvc_recording(self, _checked=False):
        if self.dual_uvc_recording_worker is not None:
            QMessageBox.information(self, 'Recording Active', 'Dual UVC recording is already active.')
            return
        if not self.dual_uvc_frames_ready():
            QMessageBox.information(self, 'No Dual UVC Frames', 'Start both cameras and wait for Camera A and Camera B frames.')
            return
        fps_a = self.dual_uvc_actual_fps_a if 1.0 <= self.dual_uvc_actual_fps_a <= 240.0 else float(DEFAULT_UVC_FPS)
        fps_b = self.dual_uvc_actual_fps_b if 1.0 <= self.dual_uvc_actual_fps_b <= 240.0 else float(DEFAULT_UVC_FPS)
        recording_fps = min(fps_a, fps_b)
        out = self.make_output('dual_uvc_recording')
        file_tag_a, file_tag_b = self.dual_uvc_image_file_tags()
        video_path = out / 'dual_uvc_comparison.avi'
        overview_path = out / '00_dual_uvc_overview.png'
        self.save_dual_uvc_overview(overview_path)
        started_at = datetime.now().astimezone().isoformat(timespec='seconds')
        manifest = {'format_version': 1, 'mode': 'dual_uvc_recording', 'started_at': started_at, 'pairing': 'latest-frame software pairing; not hardware synchronized', **self.dual_uvc_camera_manifest(), 'recording_fps': round(recording_fps, 3), 'recording_output': video_path.name, 'overview_output': overview_path.name}
        with open(out / 'recording.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        label_a, label_b = self.dual_uvc_labels()
        worker = UvcRecordingThread(video_path, recording_fps, self, left_label=label_a, right_label=label_b)
        worker.finished.connect(self.handle_dual_uvc_recording_finished)
        worker.finished.connect(worker.deleteLater)
        self.dual_uvc_recording_worker = worker
        self.dual_uvc_recording_output_dir = out
        self.dual_uvc_recording_manifest = manifest
        self.dual_uvc_recording_stop_reason = None
        self.dual_uvc_recording_started_perf = time.perf_counter()
        self.update_dual_uvc_button_states()
        worker.start()
        worker.submit(self.dual_uvc_frame_a, self.dual_uvc_frame_b, self.dual_uvc_recording_started_perf)

    def write_dual_uvc_recording_summary(self, worker, reason):
        out = self.dual_uvc_recording_output_dir
        manifest = self.dual_uvc_recording_manifest
        if out is None or manifest is None:
            return
        stats = worker.result_summary or {}
        name_a = manifest['camera_a']['image_name']
        name_b = manifest['camera_b']['image_name']
        file_tag_a = manifest['camera_a']['image_name_file_tag']
        file_tag_b = manifest['camera_b']['image_name_file_tag']
        manifest['ended_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        manifest['stop_reason'] = reason
        manifest['success'] = worker.error_message is None and bool(stats.get('written_frames', 0))
        manifest['error'] = worker.error_message
        manifest['recording'] = stats
        with open(out / 'recording.json', 'w', encoding='utf-8') as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / 'recording_summary.txt', 'w', encoding='utf-8') as summary:
            summary.write(f"Started: {manifest['started_at']}\n")
            summary.write(f"Ended: {manifest['ended_at']}\n")
            summary.write(f'Stop reason: {reason}\n')
            summary.write(f"Success: {manifest['success']}\n")
            if worker.error_message:
                summary.write(f'Error: {worker.error_message}\n')
            summary.write(f"Images: {name_a} (index {manifest['camera_a']['camera_index']}) / {name_b} (index {manifest['camera_b']['camera_index']})\n")
            summary.write(f"Recording FPS: {manifest['recording_fps']:.3f}\n")
            summary.write(f"Duration: {stats.get('duration_seconds', 0.0):.3f} s\n")
            summary.write(f"Submitted / written / duplicate frames: {stats.get('submitted_frames', 0)} / {stats.get('written_frames', 0)} / {stats.get('duplicate_frames', 0)}\n")
            summary.write(f"Queue / timestamp dropped frames: {stats.get('queue_dropped_frames', 0)} / {stats.get('timestamp_dropped_frames', 0)}\n")
            summary.write(f"File size: {stats.get('file_size_bytes', 0)} bytes\n")
            summary.write(f"Overview: {manifest['overview_output']}\n")

    def finalize_dual_uvc_recording_worker(self, worker, reason):
        if worker is not self.dual_uvc_recording_worker:
            return
        try:
            self.write_dual_uvc_recording_summary(worker, reason)
        except Exception as exc:
            self.log.append(f'Dual UVC recording summary error: {exc}')
        out = self.dual_uvc_recording_output_dir
        video_path = None if out is None else out / 'dual_uvc_comparison.avi'
        if worker.error_message:
            self.log.append(f'Dual UVC recording error: {worker.error_message}')
        else:
            stats = worker.result_summary or {}
            self.log.append(f"Dual UVC recording stopped: {video_path}; written={stats.get('written_frames', 0)}, duplicates={stats.get('duplicate_frames', 0)}, dropped={stats.get('queue_dropped_frames', 0) + stats.get('timestamp_dropped_frames', 0)}")
        self.dual_uvc_recording_worker = None
        self.dual_uvc_recording_output_dir = None
        self.dual_uvc_recording_manifest = None
        self.dual_uvc_recording_stop_reason = None
        self.dual_uvc_recording_started_perf = 0.0
        self.update_dual_uvc_button_states()

    def stop_dual_uvc_recording(self, _checked=False, reason=None, wait_ms=5000, log_message=True):
        worker = self.dual_uvc_recording_worker
        if worker is None:
            return True
        stop_reason = reason or 'User requested'
        self.dual_uvc_recording_stop_reason = stop_reason
        worker.stop()
        self.dual_uvc_record_start_btn.setEnabled(False)
        self.dual_uvc_record_stop_btn.setEnabled(False)
        if not worker.wait(wait_ms):
            if log_message:
                self.log.append('Dual UVC recording thread is still finishing the AVI file.')
            return False
        self.finalize_dual_uvc_recording_worker(worker, stop_reason)
        return True

    def handle_dual_uvc_recording_finished(self):
        worker = self.sender()
        if worker is not self.dual_uvc_recording_worker:
            return
        reason = self.dual_uvc_recording_stop_reason
        if reason is None:
            reason = 'Recording error' if worker.error_message else 'Recording worker finished'
        self.finalize_dual_uvc_recording_worker(worker, reason)
