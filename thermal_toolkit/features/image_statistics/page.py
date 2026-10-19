"""features / image_statistics / page."""

from datetime import datetime
from pathlib import Path
import json
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.analysis.quality.api import quality_analysis_manifest
from thermal_toolkit.analysis.quality.fieldscale_iqa import image_quality_analysis, quality_text
from thermal_toolkit.analysis.statistics import raw_dn_image_analysis
from thermal_toolkit.core.qt import (
    QApplication,
    QGridLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    Qt,
)
from thermal_toolkit.io.export import save_csv_matrix, save_png, save_rgb_png
from thermal_toolkit.io.raw import IMAGE_INPUT_FILTER, IMAGE_INPUT_NOTE
from thermal_toolkit.reports.histogram import render_gray_histogram_rgb, save_histogram_excel
from thermal_toolkit.reports.quality import save_quality_excel, write_quality_summary
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox
from thermal_toolkit.ui.widgets.inputs import PathRow
from thermal_toolkit.ui.widgets.preview import PreviewLabel


class ImageStatisticsPageMixin:
    """features / image_statistics / page behavior composed by the main window."""

    def build_image_statistics_page(self):
        # Viewer
        p, f = self._form_page()
        self.viewer_file = PathRow(
            "file",
            IMAGE_INPUT_FILTER,
            "Select RAW / Image File to View",
        )
        image_input_note = QLabel(IMAGE_INPUT_NOTE)
        image_input_note.setWordWrap(True)
        image_input_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow(image_input_note)
        self.viewer_file.path_selected.connect(self.auto_preview_raw)
        self.viewer_file.edit.returnPressed.connect(self.auto_preview_raw)
        self.viewer_file.edit.textChanged.connect(lambda: self.set_image_save_ready("single", False))
        self.resolution_combo.currentIndexChanged.connect(
            self.refresh_raw_viewer_settings
        )
        self.bit_depth_combo.currentIndexChanged.connect(
            self.refresh_raw_viewer_settings
        )
        f.addRow("RAW / Image File", self.viewer_file)

        self.viewer_stats_group = CollapsibleGroupBox("RAW Statistics (DN)")
        self.viewer_stats_group.setObjectName("viewerStatsCard")
        viewer_stats_layout = QGridLayout(self.viewer_stats_group)
        viewer_stats_layout.setHorizontalSpacing(10)
        viewer_stats_layout.setVerticalSpacing(8)
        self.viewer_stat_values = {}
        stat_fields = [
            ("Pixel Count", "count"),
            ("Unique Levels", "unique_levels"),
            ("Minimum", "min"),
            ("Maximum", "max"),
            ("Mean", "mean"),
            ("Median / P50", "median"),
            ("Std Dev", "std"),
            ("Entropy", "entropy_bits"),
            ("P1", "p1"),
            ("P5", "p5"),
            ("P95", "p95"),
            ("P99", "p99"),
            ("P99 - P1", "p99_p1"),
            ("P99-P1 Usage", "p99_p1_percent"),
            ("Zero Clip", "zero_clip"),
            ("Bit Max Clip", "max_clip"),
            ("Over Bit Max", "over_max"),
        ]
        for index, (label_text, key) in enumerate(stat_fields):
            row = index // 2
            column = (index % 2) * 2
            name_label = QLabel(label_text)
            value_label = QLabel("--")
            value_label.setProperty("viewerStatValue", True)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setMinimumWidth(92)
            viewer_stats_layout.addWidget(name_label, row, column)
            viewer_stats_layout.addWidget(value_label, row, column + 1)
            self.viewer_stat_values[key] = value_label

        self.viewer_stats_note = QLabel(
            "Statistics use all source grayscale pixels (RAW: original DN). Std Dev uses ddof=0; "
            "P99-P1 Usage is relative to the source bit-depth maximum."
        )
        self.viewer_stats_note.setWordWrap(True)
        self.viewer_stats_note.setStyleSheet("color:#aeb8c2; font-weight:400; padding-top:4px;")
        viewer_stats_layout.addWidget(
            self.viewer_stats_note,
            (len(stat_fields) + 1) // 2,
            0,
            1,
            4,
        )
        f.addRow(self.viewer_stats_group)

        (
            self.raw_viewer_quality_group,
            self.raw_viewer_quality_labels,
        ) = self.create_single_quality_score_group(
            "Image Quality Metrics",
            collapsible=True,
        )
        f.addRow(self.raw_viewer_quality_group)

        self.raw_viewer_histogram_group = CollapsibleGroupBox(
            "RAW DN Histogram + CDF"
        )
        self.raw_viewer_histogram_group.setObjectName("viewerStatsCard")
        raw_viewer_histogram_layout = QVBoxLayout(
            self.raw_viewer_histogram_group
        )
        self.raw_viewer_histogram_preview = PreviewLabel(
            "RAW DN Histogram"
        )
        self.raw_viewer_histogram_preview.setMinimumSize(320, 180)
        self.raw_viewer_histogram_preview.setToolTip(
            "Double-click to open a resizable and maximizable histogram window"
        )
        self.raw_viewer_histogram_preview.double_clicked.connect(
            self.open_raw_histogram_window
        )
        raw_viewer_histogram_layout.addWidget(
            self.raw_viewer_histogram_preview
        )
        raw_histogram_note = QLabel(
            "The chart aggregates the selected Bit Depth range into 256 "
            "display bins. Statistics and percentile markers remain in "
            "original RAW DN; over-range pixels enter the rightmost chart bin "
            "and are reported separately. Double-click the chart to enlarge."
        )
        raw_histogram_note.setWordWrap(True)
        raw_histogram_note.setStyleSheet(
            "color:#aeb8c2; font-weight:400; padding-top:3px;"
        )
        raw_viewer_histogram_layout.addWidget(raw_histogram_note)
        f.addRow(self.raw_viewer_histogram_group)
        self.stack.addWidget(p)

    def auto_preview_raw(self, _path=None):
        if self.current_function_index() == 0:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.run_viewer(save_output=False)
            except Exception as exc:
                self.log.append(f"\nERROR: {exc}")
                QMessageBox.critical(self, "Error", str(exc))
            finally:
                QApplication.restoreOverrideCursor()

    def refresh_raw_viewer_settings(self, _value=None):
        if (
            self.current_function_index() == self.FUNCTIONS.index("Single Image Statistics")
            and self.viewer_file.text().strip()
        ):
            self.auto_preview_raw()

    def clear_viewer_stats(self):
        for value_label in self.viewer_stat_values.values():
            value_label.setText("--")
        self.raw_viewer_stats = None
        self.raw_viewer_histogram = None
        self.raw_viewer_histogram_rgb = None
        self.raw_viewer_quality = None
        self.raw_viewer_histogram_preview.set_array(None)
        self.update_single_quality_score_labels(
            self.raw_viewer_quality_labels,
            None,
        )

    def update_viewer_stats(self, image):
        self.viewer_stats_group.setTitle(f"Statistics ({self.raw_viewer_info['value_name']})")
        stats, histogram = raw_dn_image_analysis(image, self.raw_viewer_info['value_max'])
        histogram_rgb = render_gray_histogram_rgb(
            histogram,
            stats,
            value_max=self.raw_viewer_info['value_max'],
            show_cdf=self.app_settings['timeline'].get('raw_histogram_show_cdf', False),
            title=(
                f"{self.raw_viewer_info['bit_depth']}-bit {self.raw_viewer_info['value_name']} Histogram "
                f"(256 display bins)"
            ),
        )
        self.raw_viewer_stats = stats
        self.raw_viewer_histogram = histogram
        self.raw_viewer_histogram_rgb = histogram_rgb
        self.populate_raw_stat_labels(self.viewer_stat_values, stats)
        self.raw_viewer_histogram_preview.set_array(histogram_rgb)
        return stats

    def run_viewer(self, save_output=True):
        self.set_image_save_ready("single", False)
        self.current_original = None
        self.current_result = None
        self.preview_result.set_array(None)
        self.clear_viewer_stats()
        raw_path = self.viewer_file.text()
        raw, self.raw_viewer_info = self.load_analysis_source(raw_path)
        display = raw_to_display_u8(raw, self.raw_viewer_info['value_max'])
        self.raw_viewer_quality = image_quality_analysis(raw, self.raw_viewer_info['value_max'])
        self.update_single_quality_score_labels(
            self.raw_viewer_quality_labels,
            self.raw_viewer_quality,
        )
        self.current_original = raw; self.current_result = raw
        self.preview_result.set_array(display)
        self.update_image_preview_titles()
        stats = self.update_viewer_stats(raw)
        self.set_image_save_ready("single", True)
        self.log.append(f"\n[{self.raw_viewer_info['source_type'].title()}] File: {raw_path}")
        self.log.append(
            f"Resolution: {raw.shape[1]} x {raw.shape[0]}; Pixels: {raw.size:,}; "
            f"Source bit depth: {self.raw_viewer_info['bit_depth']}-bit"
        )
        self.log_stats(self.raw_viewer_info["source_type"].title(), raw, stats)
        self.log.append(f"P1={stats['p1']:.3f}  P99={stats['p99']:.3f}")
        self.log.append(quality_text(self.raw_viewer_quality))
        self.log.append(f"Preview scale: fixed 0..{self.raw_viewer_info['value_max']} {self.raw_viewer_info['value_name']}")
        if not save_output:
            return

        out = self.make_output("raw_viewer_analysis")
        preview_path = out / "raw_preview.png"
        histogram_path = out / "raw_histogram.png"
        excel_path = out / "raw_analysis.xlsx"
        quality_excel_path = out / "quality_analysis.xlsx"
        display_path = out / "display.png"
        gradient_path = out / "gradient.png"
        manifest_path = out / "analysis.json"
        summary_path = out / "analysis_summary.txt"
        save_png(preview_path, display)
        raw_csv_path = None
        if self.raw_viewer_info["source_type"] == "raw":
            raw_csv_path = out / "original_raw_dn.csv"
            save_csv_matrix(raw_csv_path, raw, "%d")
            self.log.append(f"Saved original RAW DN CSV: {raw_csv_path}")
        save_rgb_png(histogram_path, self.raw_viewer_histogram_rgb)
        save_png(display_path, self.raw_viewer_quality["display_u8"])
        save_png(gradient_path, self.raw_viewer_quality["gradient_u8"])
        manifest = {
            "format_version": 1,
            "mode": "raw_viewer_analysis",
            "analyzed_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "source_info": self.raw_viewer_info,
            "source_raw": str(Path(raw_path).resolve()),
            "resolution": {"width": raw.shape[1], "height": raw.shape[0]},
            "bit_depth": self.raw_viewer_info['bit_depth'],
            "raw_max": self.raw_viewer_info['value_max'],
            "statistics": dict(stats),
            "quality": quality_analysis_manifest(self.raw_viewer_quality),
            "preview_output": preview_path.name,
            "raw_csv_output": (
                raw_csv_path.name if raw_csv_path is not None else None
            ),
            "histogram_output": histogram_path.name,
            "histogram_display_bins": 256,
            "excel_output": excel_path.name,
            "quality_excel_output": quality_excel_path.name,
            "quality_layer_outputs": {
                "display": display_path.name,
                "gradient": gradient_path.name,
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(summary_path, "w", encoding="utf-8") as summary:
            summary.write(f"Analyzed: {manifest['analyzed_at']}\n")
            summary.write(f"Source RAW: {manifest['source_raw']}\n")
            summary.write(
                f"Resolution: {raw.shape[1]}x{raw.shape[0]}\n"
            )
            summary.write(
                f"Bit depth / max: {self.raw_viewer_info['bit_depth']} / {self.raw_viewer_info['value_max']}\n"
            )
            self.write_raw_viewer_analysis_summary(summary, stats, self.raw_viewer_info["value_name"])
            write_quality_summary(
                summary,
                "Single Image Statistics",
                self.raw_viewer_quality,
            )
            summary.write(f"Preview: {preview_path.name}\n")
            summary.write(f"Histogram: {histogram_path.name}\n")
            summary.write(f"Excel analysis: {excel_path.name}\n")
            summary.write(f"Quality analysis: {quality_excel_path.name}\n")
        save_histogram_excel(
            excel_path,
            "Single Image Statistics Analysis",
            {
                key: value
                for key, value in manifest.items()
                if key not in {"statistics", "quality"}
            },
            [
                {
                    "label": "RAW",
                    "stats": stats,
                    "histogram": self.raw_viewer_histogram,
                }
            ],
            value_max=self.raw_viewer_info['value_max'],
            value_name=self.raw_viewer_info["value_name"],
        )
        save_quality_excel(
            quality_excel_path,
            "Single Image Statistics Quality",
            {
                key: value
                for key, value in manifest.items()
                if key not in {"statistics", "quality"}
            },
            [
                {
                    "label": "Single Image Statistics",
                    "metrics": self.raw_viewer_quality["metrics"],
                    "unavailable_reasons": self.raw_viewer_quality.get("unavailable_reasons", {}),
                }
            ],
        )
        self.log.append(f"Saved RAW analysis: {manifest_path}")
        self.log.append(f"Saved RAW histogram: {histogram_path}")
        self.log.append(f"Saved RAW Excel analysis: {excel_path}")
        self.log.append(f"Saved image quality analysis: {quality_excel_path}")
