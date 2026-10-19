"""features / temporal_drift / worker."""

from thermal_toolkit.core.qt import QThread, pyqtSignal
from thermal_toolkit.features.temporal_drift.analysis import analyze_temporal_drift


class TemporalDriftAnalysisThread(QThread):
    progress = pyqtSignal(int, int, int, int, int)
    result_ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id, source, parent=None):
        super().__init__(parent)
        self.request_id = request_id
        self.source = source

    def run(self):
        try:
            result = analyze_temporal_drift(
                *self.source,
                progress=lambda *counts: self.progress.emit(self.request_id, *counts),
                cancelled=self.isInterruptionRequested,
            )
            if result is not None and not self.isInterruptionRequested():
                self.result_ready.emit(self.request_id, result)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(self.request_id, str(exc))
