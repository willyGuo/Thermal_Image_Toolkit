"""features / temporal_drift / page."""

from pathlib import Path
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.core.qt import QLabel, QTimer, Qt
from thermal_toolkit.features.temporal_drift.export import save_temporal_drift_results
from thermal_toolkit.features.temporal_drift.help_text import TEMPORAL_DRIFT_FIELD_HELP, TEMPORAL_DRIFT_GROUP_HELP
from thermal_toolkit.features.temporal_drift.worker import TemporalDriftAnalysisThread
from thermal_toolkit.ui.widgets.inputs import PathRow


class TemporalDriftPageMixin:
    """features / temporal_drift / page behavior composed by the main window."""

    def build_temporal_drift_page(self):
        # Temporal Drift
        p, f = self._form_page()
        self._drift_worker = None
        self._drift_request_id = 0
        self._drift_result = None
        self._drift_result_source = None
        self._drift_needs_analysis = False
        self._drift_closing = False
        self.drift_folder = PathRow("folder", dialog_title="Select RAW Folder for Temporal Drift Analysis")
        f.addRow("RAW Folder", self.drift_folder)
        f.labelForField(self.drift_folder).setToolTip(TEMPORAL_DRIFT_FIELD_HELP["folder"])
        self.drift_folder.edit.setToolTip(TEMPORAL_DRIFT_FIELD_HELP["folder"])
        self.drift_status = QLabel("Select a RAW folder to analyze.")
        self.drift_status.setWordWrap(True)
        self.drift_status.setTextFormat(Qt.PlainText)
        f.addRow(self.drift_status)
        self.drift_stat_values = {}
        for title, fields in (
            ("Dataset", (("Valid Frames", "valid_frames"), ("Skipped Files", "skipped_files"),
                         ("Resolution", "resolution"), ("Bit Depth", "bit_depth"))),
            ("Frame Mean (DN)", (("First Frame", "first"), ("Last Frame", "last"),
                                 ("Last - First", "change"), ("Peak-to-Peak", "frame_p2p"),
                                 ("Mean", "frame_mean"), ("Temporal Std", "frame_std"))),
            ("Pixel Temporal Std (DN)", (("Mean", "std_mean"), ("P95", "std_p95"), ("Maximum", "std_max"))),
            ("Pixel Peak-to-Peak (DN)", (("Mean", "p2p_mean"), ("P95", "p2p_p95"), ("Maximum", "p2p_max"))),
        ):
            group, grid = self.create_quality_score_container(title)
            group.setToolTip(TEMPORAL_DRIFT_GROUP_HELP[title])
            for row, (label, key) in enumerate(fields):
                name = QLabel(label)
                value = QLabel("--")
                name.setToolTip(TEMPORAL_DRIFT_FIELD_HELP[key])
                value.setToolTip(TEMPORAL_DRIFT_FIELD_HELP[key])
                value.setProperty("viewerStatValue", True)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(name, row, 0)
                grid.addWidget(value, row, 1)
                self.drift_stat_values[key] = value
            f.addRow(group)
        self.drift_help = QLabel(
            "Temporal Std：每個像素隨時間的整體波動程度。\n"
            "Peak-to-Peak：每個像素隨時間的最大 DN - 最小 DN。\n"
            "P95：95% 像素的指標不超過此值；Std 使用 ddof=0。\n"
            "依檔名排序，影格序號不代表秒數。\n"
            "建議使用固定溫度、固定場景且沒有 FFC／Shutter 動作的 RAW。\n"
            "Save 會儲存分析資料、趨勢圖、分布圖及附 DN 色階的熱圖。"
        )
        self.drift_help.setWordWrap(True)
        self.drift_help.setStyleSheet("color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; border-radius:5px; padding:8px;")
        f.addRow(self.drift_help)
        self.drift_preview_timer = QTimer(self)
        self.drift_preview_timer.setSingleShot(True)
        self.drift_preview_timer.setInterval(250)
        self.drift_preview_timer.timeout.connect(self.auto_preview_drift)
        self.drift_folder.edit.textChanged.connect(self.schedule_drift_preview)
        self.drift_folder.path_selected.connect(self.schedule_drift_preview)
        self.drift_folder.edit.returnPressed.connect(self.schedule_drift_preview)
        self.resolution_combo.currentIndexChanged.connect(self.schedule_drift_preview)
        self.bit_depth_combo.currentIndexChanged.connect(self.schedule_drift_preview)
        self.stack.addWidget(p)

    def drift_source_key(self):
        folder = self.drift_folder.text()
        return (str(Path(folder).expanduser().resolve()) if folder else "",
                self.width, self.height, self.bit_depth)

    def drift_page_active(self):
        return self.FUNCTIONS[self.current_function_index()] == "Temporal Drift Analyzer"

    def schedule_drift_preview(self, *_args):
        self._drift_request_id += 1
        self._drift_result = None
        self._drift_result_source = None
        self._drift_needs_analysis = bool(self.drift_folder.text())
        self.drift_preview_timer.stop()
        if self._drift_worker is not None:
            self._drift_worker.requestInterruption()
        for key, label in self.drift_stat_values.items():
            label.setText("--")
            label.setToolTip(TEMPORAL_DRIFT_FIELD_HELP[key])
        self.drift_status.setText("Waiting to analyze..." if self._drift_needs_analysis else "Select a RAW folder to analyze.")
        if self.drift_page_active():
            self.run_btn.setEnabled(False)
            self.current_original = self.current_result = None
            self.preview_result.set_array(None)
            if self._drift_needs_analysis and not self._drift_closing:
                self.drift_preview_timer.start()

    def auto_preview_drift(self):
        if (self._drift_closing or not self.drift_page_active()
                or not self._drift_needs_analysis or self._drift_worker is not None):
            return
        self._drift_needs_analysis = False
        self.drift_status.setText("Scanning RAW folder...")
        worker = TemporalDriftAnalysisThread(self._drift_request_id, self.drift_source_key(), self)
        self._drift_worker = worker
        worker.progress.connect(self.update_drift_progress)
        worker.result_ready.connect(self.receive_drift_result)
        worker.failed.connect(self.drift_analysis_failed)
        worker.finished.connect(self.drift_worker_finished)
        worker.start()

    def update_drift_progress(self, request_id, processed, total, valid, skipped):
        if request_id == self._drift_request_id and not self._drift_closing:
            self.drift_status.setText(f"Analyzing {processed:,} / {total:,} files | valid {valid:,} | skipped {skipped:,}")

    def receive_drift_result(self, request_id, result):
        if request_id != self._drift_request_id or self._drift_closing:
            return
        self._drift_result = result
        self._drift_result_source = self.drift_source_key()
        self.show_drift_result()

    def drift_analysis_failed(self, request_id, message):
        if request_id == self._drift_request_id and not self._drift_closing:
            self.drift_status.setText("Analysis error: " + message)
            if self.drift_page_active():
                self.run_btn.setEnabled(False)

    def drift_worker_finished(self):
        worker = self.sender()
        if worker is self._drift_worker:
            self._drift_worker = None
        worker.deleteLater()
        if (self._drift_needs_analysis and self.drift_page_active()
                and not self._drift_closing and not self.drift_preview_timer.isActive()):
            self.drift_preview_timer.start()

    def show_drift_result(self):
        result = self._drift_result
        if result is None:
            return
        report = result["report"]
        values = {"valid_frames": f"{report['valid_frames']:,}",
                  "skipped_files": f"{len(report['skipped_files']):,}",
                  "resolution": f"{report['width']} x {report['height']}",
                  "bit_depth": str(report["bit_depth"])}
        for key, metric in (("first", "first"), ("last", "last"), ("change", "change"),
                            ("frame_p2p", "peak_to_peak"), ("frame_mean", "mean"), ("frame_std", "std")):
            values[key] = format(report["frame_mean"][metric], "+.3f" if key == "change" else ".3f")
        for prefix, metric in (("std", "pixel_temporal_std"), ("p2p", "pixel_peak_to_peak")):
            for stat in ("mean", "p95", "max"):
                values[prefix + "_" + stat] = f"{report[metric][stat]:.3f}"
        for key, text in values.items():
            self.drift_stat_values[key].setText(text)
        status = f"Ready | {report['valid_frames']:,} valid / {report['file_count']:,} files | Save to export."
        if report["valid_frames"] == 1:
            status += "\nOnly one frame: temporal variation cannot be assessed."
        self.drift_status.setText(status)
        skipped_details = "\n".join(
            row["filename"] + ": " + row["reason"] for row in report["skipped_files"][:20]
        )
        self.drift_stat_values["skipped_files"].setToolTip(
            TEMPORAL_DRIFT_FIELD_HELP["skipped_files"] + "\n\n"
            + ("本次略過的檔案（最多顯示 20 筆）：\n" + skipped_details
               if skipped_details else "本次沒有略過檔案。")
        )
        if not self.drift_page_active():
            return
        self.current_original = result["first_frame"]
        self.current_result = result["pixel_std"]
        self.preview_result.set_array(raw_to_display_u8(self.current_result, (1 << report["bit_depth"]) - 1))
        self.run_btn.setEnabled(True)

    def run_drift(self):
        if self._drift_result is None or self._drift_result_source != self.drift_source_key():
            raise ValueError("Wait for the current folder analysis to finish before saving.")
        out = self.make_output("temporal_drift")
        report = save_temporal_drift_results(out, self._drift_result)
        self.log.append(f"Saved cached analysis: {report['valid_frames']:,} frames; {len(report['skipped_files']):,} skipped.")
        for filename in report["charts"]:
            self.log.append(f"Saved chart: {out / filename}")
        self.log.append(f"Saved summary: {out / 'temporal_drift_summary.json'}")
