"""features / image_algorithm / statistics."""

from thermal_toolkit.analysis.statistics import raw_dn_image_analysis
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import (
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
    Qt,
)
from thermal_toolkit.reports.histogram import render_gray_histogram_rgb
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox
from thermal_toolkit.ui.widgets.preview import PreviewLabel, UvcVideoWindow


class ImageAlgorithmStatisticsMixin:
    """features / image_algorithm / statistics behavior composed by the main window."""

    def create_raw_algorithm_statistics(self, f):
        self.algorithm_stats_group = CollapsibleGroupBox("RAW Statistics (DN)")
        self.algorithm_stats_group.setObjectName("viewerStatsCard")
        algorithm_stats_layout = QGridLayout(self.algorithm_stats_group)
        algorithm_stats_layout.setHorizontalSpacing(10)
        algorithm_stats_layout.setVerticalSpacing(8)
        self.algorithm_stat_values = {}
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
            algorithm_stats_layout.addWidget(name_label, row, column)
            algorithm_stats_layout.addWidget(value_label, row, column + 1)
            self.algorithm_stat_values[key] = value_label

        self.algorithm_stats_note = QLabel(
            "Statistics use all source grayscale pixels (RAW: original DN). Std Dev uses ddof=0; "
            "P99-P1 Usage is relative to the source bit-depth maximum."
        )
        self.algorithm_stats_note.setWordWrap(True)
        self.algorithm_stats_note.setStyleSheet("color:#aeb8c2; font-weight:400; padding-top:4px;")
        algorithm_stats_layout.addWidget(
            self.algorithm_stats_note,
            (len(stat_fields) + 1) // 2,
            0,
            1,
            4,
        )
        self.algorithm_statistics_source = FocusWheelComboBox()
        self.algorithm_statistics_source.addItem("Original RAW (DN)", "raw")
        self.algorithm_statistics_source.addItem("Final Pipeline Output (8-bit)", "final")
        self.algorithm_statistics_source_row = QWidget()
        source_layout = QVBoxLayout(self.algorithm_statistics_source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel("Statistics Source"))
        source_layout.addWidget(self.algorithm_statistics_source)
        algorithm_stats_layout.addWidget(
            self.algorithm_statistics_source_row, (len(stat_fields) + 1) // 2 + 1, 0, 1, 4
        )
        self.algorithm_statistics_source_row.hide()
        self.algorithm_statistics_source.currentIndexChanged.connect(
            lambda _index: self.update_raw_algorithm_statistics(self.raw_algorithm_source)
        )
        f.addRow(self.algorithm_stats_group)
        self.raw_algorithm_histogram_group = CollapsibleGroupBox(
            "RAW DN Histogram + CDF"
        )
        self.raw_algorithm_histogram_group.setObjectName("viewerStatsCard")
        raw_algorithm_histogram_layout = QVBoxLayout(
            self.raw_algorithm_histogram_group
        )
        self.raw_algorithm_histogram_preview = PreviewLabel(
            "RAW DN Histogram"
        )
        self.raw_algorithm_histogram_preview.setMinimumSize(320, 180)
        self.raw_algorithm_histogram_preview.setToolTip(
            "Double-click to open a resizable and maximizable histogram window"
        )
        self.raw_algorithm_histogram_preview.double_clicked.connect(
            self.open_algorithm_histogram_window
        )
        raw_algorithm_histogram_layout.addWidget(
            self.raw_algorithm_histogram_preview
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
        raw_algorithm_histogram_layout.addWidget(raw_histogram_note)
        f.addRow(self.raw_algorithm_histogram_group)
        self.raw_algorithm_histogram_rgb = None
        self.raw_algorithm_stats = None

    def clear_raw_algorithm_statistics(self):
        if not hasattr(self, "algorithm_stat_values"):
            return
        for label in self.algorithm_stat_values.values():
            label.setText("--")
        self.raw_algorithm_stats = None
        self.raw_algorithm_histogram_rgb = None
        self.raw_algorithm_histogram_preview.set_array(None)
        self.close_raw_histogram_window()

    def update_raw_algorithm_statistics(self, image):
        if image is None:
            self.clear_raw_algorithm_statistics()
            self.algorithm_statistics_source_row.hide()
            return
        info = self.raw_algorithm_info
        native_input = info["bit_depth"] > 8
        self.algorithm_statistics_source_row.setVisible(native_input)
        if not native_input:
            self.algorithm_statistics_source.blockSignals(True)
            self.algorithm_statistics_source.setCurrentIndex(0)
            self.algorithm_statistics_source.blockSignals(False)
        use_final = native_input and self.algorithm_statistics_source.currentData() == "final"
        if use_final:
            image = self.raw_pipeline_final
            info = {"bit_depth": 8, "value_max": 255, "value_name": "Final Output Gray"}
        self.raw_algorithm_statistics_info = info
        self.algorithm_stats_group.setTitle(
            "Final Pipeline Statistics (8-bit)" if use_final else "RAW Statistics (DN)"
        )
        self.algorithm_stats_note.setText(
            "Statistics use final pipeline output (0..255). Std Dev uses ddof=0."
            if use_final else
            "Statistics use all original source pixels. Std Dev uses ddof=0; "
            "P99-P1 Usage is relative to the source bit-depth maximum."
        )
        if image is None:
            self.clear_raw_algorithm_statistics()
            return
        stats, histogram = raw_dn_image_analysis(image, info["value_max"])
        self.raw_algorithm_stats = stats
        self.raw_algorithm_histogram = histogram
        self.populate_raw_stat_labels(self.algorithm_stat_values, stats)
        self.raw_algorithm_histogram_rgb = render_gray_histogram_rgb(
            histogram, stats, value_max=info["value_max"],
            show_cdf=self.app_settings["timeline"].get("raw_histogram_show_cdf", False),
            title=f"{info['bit_depth']}-bit {info['value_name']} Histogram (256 display bins)",
        )
        self.raw_algorithm_histogram_preview.set_array(self.raw_algorithm_histogram_rgb)
        self.raw_algorithm_histogram_group.setTitle(
            info["value_name"] + " Histogram"
            + (" + CDF" if self.app_settings["timeline"].get("raw_histogram_show_cdf", False) else "")
        )
        if self.raw_viewer_histogram_window is not None and self.raw_viewer_histogram_window.isVisible():
            self.open_algorithm_histogram_window()

    def open_algorithm_histogram_window(self):
        if self.raw_algorithm_histogram_rgb is None:
            return
        if self.raw_viewer_histogram_window is None:
            self.raw_viewer_histogram_window = UvcVideoWindow(self)
        window = self.raw_viewer_histogram_window
        window.setWindowTitle("Image Algorithm Compare - DN Histogram and CDF")
        window.set_bgr_frame(cv2.cvtColor(self.raw_algorithm_histogram_rgb, cv2.COLOR_RGB2BGR))
        window.show()
        window.raise_()
        window.activateWindow()
