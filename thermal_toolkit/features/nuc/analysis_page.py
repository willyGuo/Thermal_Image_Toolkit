"""features / nuc / analysis_page."""

from pathlib import Path
import csv
import json
import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.core.constants import RAW_DTYPE
from thermal_toolkit.core.qt import QLabel, QTimer, Qt
from thermal_toolkit.features.nuc.analysis import nuc_analysis_statistics, nuc_roi_analysis
from thermal_toolkit.features.nuc.calibration import apply_novatek_nuc, load_novatek_nuc_bin
from thermal_toolkit.features.nuc.definitions import (
    NUC_ANALYSIS_FIELDS,
    NUC_ANALYSIS_HELP,
    NUC_CHANGE_HELP,
    NUC_COMPARISON_ROIS,
    NUC_ROI_METRICS,
)
from thermal_toolkit.features.nuc.export import save_nuc_analysis_xlsx, save_nuc_statistics_xlsx
from thermal_toolkit.io.export import save_png
from thermal_toolkit.io.raw import load_raw
from thermal_toolkit.ui.widgets.inputs import FocusWheelSpinBox, PathRow


class NucAnalysisPageMixin:
    """features / nuc / analysis_page behavior composed by the main window."""

    def build_nuc_analysis_page(self):
        # NUC_Analyze
        p, f = self._form_page()
        self.k_bin_file = PathRow("file", "K Table (*.bin *.BIN)", "Select NUC K Table")
        self.b_bin_file = PathRow("file", "B Table (*.bin *.BIN)", "Select NUC B Table")
        self.apply_raw_file = PathRow("file", "RAW Files (*.raw *.RAW *.bin *.BIN)", "Select RAW for NUC_Analyze")
        self.nuc_gain_shift = FocusWheelSpinBox()
        self.nuc_gain_shift.setRange(0, 31)
        self.nuc_gain_shift.setValue(8)
        self.nuc_gain_shift.setToolTip("縮放位數 N：輸出＝K×（輸入－B）÷2^N，預設為 8。先向下取整，再限制到所選位深範圍；N 必須與 K 表的定點格式相符。")
        f.addRow("K Table (uint16 LE)", self.k_bin_file)
        f.addRow("B Table (int16 LE)", self.b_bin_file)
        f.addRow("RAW File", self.apply_raw_file)
        f.addRow("N", self.nuc_gain_shift)
        note = QLabel("Output = K * (Input - B) / 2^N\nInteger floor, then clip to Bit Depth range.\nSpatial Std uses all pixels (ddof=0). Reduction = (original - corrected) / original * 100%; negative means increase, N/A means original is zero.")
        note.setWordWrap(True)
        f.addRow(note)
        self.nuc_stats_group, stats_layout = self.create_quality_score_container("NUC Statistics")
        stats_layout.setHorizontalSpacing(10)
        stats_layout.setVerticalSpacing(7)
        for column, header in enumerate(("Metric", "Original", "Corrected")):
            label = QLabel(header)
            label.setToolTip((
                "統計指標：將滑鼠停在欄位名稱或數值上，可查看中文定義。DN 是原始數位值，不是溫度。",
                "原始 RAW：套用本次 K、B 校正前的統計。",
                "校正後 RAW：套用 K、B 與 N，向下取整並限制到所選位深範圍後的統計。",
            )[column])
            label.setStyleSheet("font-weight:700; color:#c9e2f2;")
            label.setAlignment(Qt.AlignLeft if column == 0 else Qt.AlignRight)
            stats_layout.addWidget(label, 0, column)
        self.nuc_stat_labels = {}
        fields = tuple((title, key) for title, key, unit in NUC_ANALYSIS_FIELDS)
        for row, (title, key) in enumerate(fields, 1):
            metric_label = QLabel(title)
            metric_label.setWordWrap(True)
            metric_label.setToolTip(NUC_ANALYSIS_HELP[key])
            stats_layout.addWidget(metric_label, row, 0)
            values = []
            for column in (1, 2):
                value = QLabel("--")
                value.setProperty("viewerStatValue", True)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setMinimumWidth(66)
                value.setToolTip(NUC_ANALYSIS_HELP[key])
                stats_layout.addWidget(value, row, column)
                values.append(value)
            self.nuc_stat_labels[key] = values
        self.nuc_change_labels = {}
        changes = (("Std reduction", "std"), ("Peak-to-Peak reduction", "peak_to_peak"),
                   ("Mean change", "mean"), ("Clipped low / high", "clipped"), ("Pixels", "pixels"))
        for row, (title, key) in enumerate(changes, len(fields) + 1):
            change_label = QLabel(title)
            change_label.setToolTip(NUC_CHANGE_HELP[key])
            stats_layout.addWidget(change_label, row, 0)
            value = QLabel("--")
            value.setToolTip(NUC_CHANGE_HELP[key])
            value.setProperty("viewerStatValue", True)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            stats_layout.addWidget(value, row, 1, 1, 2)
            self.nuc_change_labels[key] = value
        self.nuc_statistics = QLabel("Select K, B and RAW files.")
        self.nuc_statistics.setWordWrap(True)
        self.nuc_statistics.setTextFormat(Qt.PlainText)
        self.nuc_statistics.setStyleSheet("color:#aeb8c2; font-weight:400; padding-top:4px;")
        stats_layout.addWidget(self.nuc_statistics, len(fields) + len(changes) + 1, 0, 1, 3)
        f.addRow(self.nuc_stats_group)
        self.nuc_roi_labels = {}
        self.nuc_roi_reduction_labels = {}
        for name, coordinates in (*NUC_COMPARISON_ROIS.items(), ("左右合併參考", (0, 0, 0, 0))):
            x0, y0, x1, y1 = coordinates
            group, grid = self.create_quality_score_container(name + " Statistics")
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(7)
            for column, title in enumerate(("Metric", "Original", "Corrected")):
                label = QLabel(title)
                label.setStyleSheet("font-weight:700; color:#c9e2f2;")
                label.setAlignment(Qt.AlignLeft if column == 0 else Qt.AlignRight)
                grid.addWidget(label, 0, column)
            for row, key in enumerate(NUC_ROI_METRICS, 1):
                row = row * 2 - 1
                title = next(title for title, metric, _ in NUC_ANALYSIS_FIELDS if metric == key)
                label = QLabel(title)
                label.setWordWrap(True)
                label.setToolTip("僅對本區域計算。" + NUC_ANALYSIS_HELP[key])
                grid.addWidget(label, row, 0)
                values = []
                for column in (1, 2):
                    value = QLabel("--")
                    value.setProperty("viewerStatValue", True)
                    value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    value.setMinimumWidth(66)
                    value.setToolTip("僅對本區域計算。" + NUC_ANALYSIS_HELP[key])
                    grid.addWidget(value, row, column)
                    values.append(value)
                self.nuc_roi_labels[name, key] = values
                reduction_label = QLabel("下降：--")
                reduction_label.setWordWrap(True)
                reduction_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                reduction_label.setToolTip("下降量＝原始－NUC後；下降率＝下降量÷原始×100%。負值表示增加；原始為零時百分比為 N/A。單位與上方指標一致。平均亮度下降不一定表示校正改善。")
                grid.addWidget(reduction_label, row + 1, 0, 1, 3)
                self.nuc_roi_reduction_labels[name, key] = reduction_label
            note = QLabel(f"X=[{x0},{x1}), Y=[{y0},{y1})\n{x1-x0} x {y1-y0} pixels；終點不包含。")
            if name == "左右合併參考":
                note.setText("合併左右兩區共 1,728 像素，不包含中間方塊。平均值等於左右平均；標準差由合併像素計算，並非兩個標準差的平均。")
            note.setWordWrap(True)
            grid.addWidget(note, len(NUC_ROI_METRICS) * 2 + 1, 0, 1, 3)
            f.addRow(group)
        self.nuc_roi_comparison = QLabel("--")
        self.nuc_roi_comparison.setWordWrap(True)
        self.nuc_roi_comparison.setToolTip("左右參考平均＝（左側平均＋右側平均）÷2。方塊平均減左右參考平均，正值表示方塊較亮，負值表示較暗。")
        f.addRow(self.nuc_roi_comparison)
        self.nuc_preview_timer = QTimer(self)
        self.nuc_preview_timer.setSingleShot(True)
        self.nuc_preview_timer.setInterval(180)
        self.nuc_preview_timer.timeout.connect(self.auto_preview_nuc)
        self.k_bin_file.path_selected.connect(self.auto_pair_nuc_b_file)
        for row in (self.k_bin_file, self.b_bin_file, self.apply_raw_file):
            row.edit.textChanged.connect(self.schedule_nuc_preview)
        self.nuc_gain_shift.valueChanged.connect(self.schedule_nuc_preview)
        self.resolution_combo.currentIndexChanged.connect(self.schedule_nuc_preview)
        self.bit_depth_combo.currentIndexChanged.connect(self.schedule_nuc_preview)
        self.stack.addWidget(p)

    def auto_pair_nuc_b_file(self, k_path):
        path = Path(k_path)
        if path.name.lower().endswith("_k.bin"):
            candidate = path.with_name(f"{path.name[:-6]}_B.bin")
            if candidate.is_file():
                self.b_bin_file.setText(str(candidate))

    def reset_nuc_statistics(self, message):
        for label in self.nuc_roi_reduction_labels.values():
            label.setText("下降：--")
        for values in self.nuc_roi_labels.values():
            for label in values:
                label.setText("--")
        self.nuc_roi_comparison.setText("--")
        for values in self.nuc_stat_labels.values():
            for label in values:
                label.setText("--")
        for label in self.nuc_change_labels.values():
            label.setText("--")
        self.nuc_statistics.setText(message)

    def schedule_nuc_preview(self, *_args):
        if self.current_function_index() == 3:
            self.run_btn.setEnabled(False)
            self.reset_nuc_statistics("Updating...")
            self.preview_original.set_array(None)
            self.preview_result.set_array(None)
            self.current_original = self.current_result = None
            self.nuc_preview_timer.start()

    def auto_preview_nuc(self):
        if self.current_function_index() != 3:
            return
        self.run_btn.setEnabled(False)
        self.current_original = self.current_result = None
        self.preview_original.set_array(None)
        self.preview_result.set_array(None)
        if not all(row.text().strip() for row in (self.k_bin_file, self.b_bin_file, self.apply_raw_file)):
            self.reset_nuc_statistics("Select K, B and RAW files.")
            return
        try:
            self.run_nuc_analyze(save_output=False)
        except Exception as exc:
            self.reset_nuc_statistics("Analysis error: " + str(exc))

    def run_nuc_analyze(self, save_output=True):
        raw = load_raw(self.apply_raw_file.text(), self.width, self.height)
        stored_k = load_novatek_nuc_bin(self.k_bin_file.text(), self.width, self.height, "<u2", "K")
        gfx_b = load_novatek_nuc_bin(self.b_bin_file.text(), self.width, self.height, "<i2", "B")
        n = self.nuc_gain_shift.value()
        unbounded = (stored_k.astype(np.int64) * (raw.astype(np.int64) - gfx_b.astype(np.int64))) >> n
        corrected = apply_novatek_nuc(raw, stored_k, gfx_b, self.raw_max, n)
        before = nuc_analysis_statistics(raw, self.raw_max)
        after = nuc_analysis_statistics(corrected, self.raw_max)
        roi_analysis = nuc_roi_analysis(raw, corrected, stored_k, gfx_b, self.raw_max, n)
        for name, region in roi_analysis["regions"].items():
            for key in NUC_ROI_METRICS:
                reduction = region["reduction"][key]
                amount = "N/A" if reduction["amount"] is None else f"{reduction['amount']:.3f}"
                percent = "N/A" if reduction["percent"] is None else f"{reduction['percent']:.3f}%"
                self.nuc_roi_reduction_labels[name, key].setText(f"下降：{amount}（{percent}）")
                for label, stage in zip(self.nuc_roi_labels[name, key], ("original", "corrected")):
                    value = region.get(stage, {}).get(key)
                    label.setText("N/A" if value is None else (f"{value:,}" if isinstance(value, int) else f"{value:.3f}"))
        comparison = roi_analysis["comparison"]
        if comparison:
            self.nuc_roi_comparison.setText("\n".join(
                f"{title}：左右平均 {comparison[stage]['reference_mean_DN']:.3f} DN；方塊減左右平均 {comparison[stage]['block_minus_reference_DN']:+.3f} DN"
                for stage, title in (("original", "原始 RAW"), ("corrected", "NUC 後"))))
            for key, title in (("reference_mean_DN", "左右平均下降"), ("absolute_block_difference_DN", "方塊與左右亮度差縮小（絕對值）")):
                reduction = comparison["reduction"][key]
                percent = "N/A" if reduction["percent"] is None else f"{reduction['percent']:.3f}%"
                self.nuc_roi_comparison.setText(self.nuc_roi_comparison.text() + f"\n{title}：{reduction['amount']:.3f} DN（{percent}）")
        else:
            self.nuc_roi_comparison.setText("N/A：方塊或參考區超出影像範圍。")
        rows = []
        for metric in before:
            original, result = before[metric], after[metric]
            decrease = original - result if original is not None and result is not None else None
            percent = 100.0 * decrease / original if decrease is not None and original != 0 else None
            rows.append((metric, original, result, decrease, percent))
        clipped_low = int(np.count_nonzero(unbounded < 0))
        clipped_high = int(np.count_nonzero(unbounded > self.raw_max))
        for metric, original, result, decrease, percent in rows:
            original_label, corrected_label = self.nuc_stat_labels[metric]
            unit = next(unit for _, key, unit in NUC_ANALYSIS_FIELDS if key == metric)
            for label, value in ((original_label, original), (corrected_label, result)):
                label.setText("N/A" if value is None else (f"{value:,}" if unit == "count" else f"{value:.3f}"))
                label.setToolTip(NUC_ANALYSIS_HELP[metric])
            if metric in ("std", "peak_to_peak"):
                ratio = "N/A" if percent is None else f"{percent:.3f}%"
                self.nuc_change_labels[metric].setText(f"{decrease:.3f} DN ({ratio})")
        self.nuc_change_labels["mean"].setText(f"{after['mean'] - before['mean']:+.3f} DN")
        self.nuc_change_labels["clipped"].setText(f"{clipped_low:,} / {clipped_high:,}")
        self.nuc_change_labels["pixels"].setText(f"{raw.size:,}")
        self.nuc_statistics.setText("DN except pixel / level counts. Center: centered 256x256 (limited by image size); rim: 32 pixels. Blocks: complete 64x64 tiles from top-left, partial edges excluded. Residual: RAW - 3x3 median, excluding 2-pixel border; sigma = 1.4826 * MAD; candidate threshold = max(8 * sigma, 10 DN). Candidates are not confirmed bad pixels.")
        self.current_original, self.current_result = raw, corrected
        self.preview_original.set_array(raw_to_display_u8(raw, self.raw_max))
        self.preview_result.set_array(raw_to_display_u8(corrected, self.raw_max))
        self.run_btn.setEnabled(True)
        if not save_output:
            return
        out = self.make_output("nuc_analyze")
        for name, image in (("original", raw), ("corrected_nuc", corrected)):
            image.astype(RAW_DTYPE).tofile(out / (name + ".raw"))
            save_png(out / (name + ".png"), raw_to_display_u8(image, self.raw_max))
        report = {"function": "NUC_Analyze", "raw_file": self.apply_raw_file.text(),
                  "k_file": self.k_bin_file.text(), "b_file": self.b_bin_file.text(),
                  "width": self.width, "height": self.height, "bit_depth": self.bit_depth,
                  "N": n, "formula": "clip(floor(K * (input - B) / 2^N), 0, 2^bit_depth - 1)",
                  "k_format": "uint16 little-endian", "b_format": "int16 little-endian",
                  "std_ddof": 0, "pixel_count": int(raw.size),
                  "metric_units": {key: unit for _, key, unit in NUC_ANALYSIS_FIELDS},
                  "analysis_definition": self.nuc_statistics.text(),
                  "roi_analysis": roi_analysis,
                  "center_roi_xyxy": [(self.width - min(256, self.width)) // 2, (self.height - min(256, self.height)) // 2,
                                      (self.width - min(256, self.width)) // 2 + min(256, self.width),
                                      (self.height - min(256, self.height)) // 2 + min(256, self.height)],
                  "block64_count": (self.width // 64) * (self.height // 64),
                  "original": before, "corrected": after,
                  "reduction_dn": {r[0]: r[3] for r in rows},
                  "reduction_percent": {r[0]: r[4] for r in rows},
                  "mean_change_dn": after["mean"] - before["mean"],
                  "clipped_low_count": clipped_low, "clipped_high_count": clipped_high}
        with open(out / "nuc_analysis.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with open(out / "nuc_statistics.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(("metric", "unit", "original", "corrected", "reduction", "reduction_percent"))
            units = {key: unit for _, key, unit in NUC_ANALYSIS_FIELDS}
            writer.writerows(tuple("N/A" if value is None else value for value in (row[0], units[row[0]], *row[1:])) for row in rows)
        save_nuc_analysis_xlsx(out / "nuc_analysis.xlsx", raw, corrected, stored_k, gfx_b, report)
        save_nuc_statistics_xlsx(out / "statistics.xlsx", raw, corrected, report)
        with open(out / "nuc_roi_statistics.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(("區域", "X起點", "Y起點", "X終點不含", "Y終點不含", "有效", "指標", "原始", "NUC後", "下降量", "下降率%"))
            for name, region in roi_analysis["regions"].items():
                for key in NUC_ROI_METRICS:
                    before_value = region.get("original", {}).get(key)
                    after_value = region.get("corrected", {}).get(key)
                    decrease = before_value - after_value if before_value is not None else None
                    percent = decrease / before_value * 100 if before_value not in (None, 0) else None
                    writer.writerow((name, *(region["xyxy"] or ("左右兩區合併", "", "", "")), region["valid"], key,
                                     *("N/A" if value is None else value for value in (before_value, after_value, decrease, percent))))
            for key in ("reference_mean_DN", "block_minus_reference_DN", "absolute_block_difference_DN"):
                before_value = comparison.get("original", {}).get(key)
                after_value = comparison.get("corrected", {}).get(key)
                writer.writerow(("區域比較", "", "", "", "", bool(comparison), key,
                                 before_value if before_value is not None else "N/A",
                                 after_value if after_value is not None else "N/A",
                                 comparison.get("reduction", {}).get(key, {}).get("amount", "N/A"),
                                 comparison.get("reduction", {}).get(key, {}).get("percent") if comparison.get("reduction", {}).get(key, {}).get("percent") is not None else "N/A"))
        self.log.append(f"NUC_Analyze saved: {out}")
