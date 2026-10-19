"""features / image_algorithm / page."""

from pathlib import Path
import json
import numpy as np
import time
from thermal_toolkit.algorithms.fieldscale import FIELDSCALE_RAW14_MAX
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.algorithms.parameters import DEFAULT_TONE_MAPPING_PARAMETERS, TONE_MAPPING_OPTIONS
from thermal_toolkit.algorithms.pipeline import apply_algorithm_pipeline, pipeline_execution_json, pipeline_record_image
from thermal_toolkit.algorithms.weighted_grid import weighted_tiles
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping, tone_mapping_parameters
from thermal_toolkit.analysis.quality.api import quality_analysis_manifest
from thermal_toolkit.analysis.quality.fieldscale_iqa import image_quality_analysis, quality_text
from thermal_toolkit.analysis.statistics import raw_dn_image_analysis
from thermal_toolkit.core.constants import DUAL_UVC_IMAGE_NAME_OPTIONS
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    Qt,
)
from thermal_toolkit.io.export import save_csv_matrix, save_png, save_rgb_png
from thermal_toolkit.io.raw import IMAGE_INPUT_FILTER, IMAGE_INPUT_NOTE
from thermal_toolkit.reports.histogram import save_histogram_excel
from thermal_toolkit.reports.histogram_weighted import save_histogram_weighted_intervals
from thermal_toolkit.reports.pipeline import save_pipeline_overview_png
from thermal_toolkit.reports.quality import save_quality_excel, write_quality_summary
from thermal_toolkit.ui.pipeline_editor import AlgorithmPipelineEditor
from thermal_toolkit.ui.weighted_mapping_details import weighted_mapping_details_html
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox, QualityScoreCheckBox
from thermal_toolkit.ui.widgets.inputs import (
    FocusWheelComboBox,
    FocusWheelDoubleSpinBox,
    FocusWheelSpinBox,
    PathRow,
)


class ImageAlgorithmPageMixin:
    """features / image_algorithm / page behavior composed by the main window."""

    def build_image_algorithm_page(self):
        # Single Image Algorithm Compare
        p, f = self._form_page()
        self.raw_algorithm_file = PathRow(
            "file",
            IMAGE_INPUT_FILTER,
            "Select RAW / Image File for Algorithm Compare",
        )
        image_input_note = QLabel(IMAGE_INPUT_NOTE)
        image_input_note.setWordWrap(True)
        image_input_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow(image_input_note)
        self.raw_input_mapping_combo = FocusWheelComboBox()
        for mapping_name, mapping_key in TONE_MAPPING_OPTIONS:
            self.raw_input_mapping_combo.addItem(mapping_name, mapping_key)
        self.raw_input_mapping_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.raw_input_mapping_note = QLabel(
            "Tone Mapping is the single native-DN to 8-bit conversion stage. "
            "All Algorithm Pipeline steps below receive its 8-bit output."
        )
        self.raw_input_mapping_note.setWordWrap(True)
        self.raw_input_mapping_note.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.raw_input_mapping_note.setStyleSheet("color:#c98;")
        self.tone_mapping_parameter_group = CollapsibleGroupBox(
            "Tone Mapping Parameters"
        )
        self.tone_mapping_parameter_group.setObjectName("pipelineParameterCard")
        self.tone_mapping_parameter_layout = QVBoxLayout(
            self.tone_mapping_parameter_group
        )
        self.tone_mapping_parameter_layout.setContentsMargins(10, 10, 10, 10)
        self.tone_mapping_parameter_layout.setSpacing(8)

        self.tone_black_dn = FocusWheelSpinBox()
        self.tone_black_dn.setRange(0, 65534)
        self.tone_black_dn.setValue(0)
        self.tone_white_dn = FocusWheelSpinBox()
        self.tone_white_dn.setRange(1, 65535)
        self.tone_white_dn.setValue(FIELDSCALE_RAW14_MAX)
        self.tone_low_percentile = FocusWheelDoubleSpinBox()
        self.tone_low_percentile.setRange(0.0, 99.5)
        self.tone_low_percentile.setDecimals(1)
        self.tone_low_percentile.setSingleStep(0.5)
        self.tone_low_percentile.setValue(1.0)
        self.tone_high_percentile = FocusWheelDoubleSpinBox()
        self.tone_high_percentile.setRange(0.5, 100.0)
        self.tone_high_percentile.setDecimals(1)
        self.tone_high_percentile.setSingleStep(0.5)
        self.tone_high_percentile.setValue(99.0)
        self.tone_percentile_gain_limit = FocusWheelDoubleSpinBox()
        self.tone_percentile_gain_limit.setRange(0.0, 4.0)
        self.tone_percentile_gain_limit.setDecimals(1)
        self.tone_percentile_gain_limit.setSingleStep(0.1)
        self.tone_percentile_gain_limit.setSpecialValueText("0.0 (Off)")
        self.tone_percentile_gain_limit.setToolTip(
            "Cap percentile stretching per native DN; 0.0 keeps the original mapping. "
            "Preserve the lower-median DN's brightness and percentile clipping without rescaling. "
            "Fractional limits apply before 8-bit rounding; later enhancements may increase contrast."
        )
        self.tone_histogram_weighted_bright_percentile = FocusWheelDoubleSpinBox()
        self.tone_histogram_weighted_bright_percentile.setRange(0.0, 100.0)
        self.tone_histogram_weighted_bright_percentile.setDecimals(1)
        self.tone_histogram_weighted_bright_percentile.setSingleStep(1.0)
        self.tone_histogram_weighted_bright_percentile.setValue(80.0)
        self.tone_histogram_weighted_bright_percentile.setToolTip(
            "Input percentile defining the bright threshold (0..100). "
            "For example, 80 uses P80 before clipping; calculated independently "
            "in each tile when Grid > 1."
        )
        self.tone_histogram_weighted_bright_weight = QComboBox()
        for weight in (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
            self.tone_histogram_weighted_bright_weight.addItem(
                "1 (Off)" if weight == 1.0 else f"{weight:g}", weight
            )
        self.tone_histogram_weighted_bright_weight.setToolTip(
            "Multiply interval weights at or above the selected input percentile, "
            "then renormalize to 0..255. 1 preserves the original mapping."
        )
        self.tone_histogram_weighted_grid = FocusWheelSpinBox()
        self.tone_histogram_weighted_gain_limit = FocusWheelDoubleSpinBox()
        self.tone_histogram_weighted_gain_limit.setRange(0.0, 4.0)
        self.tone_histogram_weighted_gain_limit.setDecimals(1)
        self.tone_histogram_weighted_gain_limit.setSingleStep(0.1)
        self.tone_histogram_weighted_gain_limit.setSpecialValueText("0.0 (Off)")
        self.tone_histogram_weighted_gain_limit.setToolTip(
            "Limit the final LUT increase per 1 native DN. Off preserves the original output. "
            "Keep the original brightness at the lower-median DN; do not stretch again. "
            "With Grid > 1, limit each tile before interpolation. This does not bound "
            "brightness differences between tiles or later enhancement steps."
            " Enter 0.0..4.0 in steps of 0.1. Fractions accumulate before rounding; "
            "8-bit output steps may exceed the fractional limit due to rounding."
        )
        self.tone_histogram_weighted_grid.setRange(1, 64)
        self.tone_histogram_weighted_grid.setValue(1)
        self.tone_histogram_weighted_grid.setToolTip(
            "每個方向的分區數：1 為全圖映射，8 為 8×8 格。每格獨立計算權重，"
            "以相鄰四格 LUT 的雙線性插值平滑銜接；無法整除時以邊緣像素補齊。"
        )
        self.tone_histogram_weighted_clip = FocusWheelDoubleSpinBox()
        self.tone_histogram_weighted_clip.setRange(0.0, 100.0)
        self.tone_histogram_weighted_clip.setDecimals(1)
        self.tone_histogram_weighted_clip.setSingleStep(0.1)
        self.tone_histogram_weighted_clip.setValue(2.0)
        self.tone_histogram_weighted_redistribute = QualityScoreCheckBox("Redistribute Clipped Pixels")
        self.tone_histogram_weighted_redistribute.setChecked(True)
        self.tone_histogram_weighted_redistribute.setToolTip(
            "Distribute excess across the full input DN range, including empty levels, below the "
            "clip count. The clipping cap is preserved. "
            "Discard any excess that cannot fit. Uncheck to discard all excess."
        )
        self.tone_histogram_weighted_clip.setToolTip(
            "0 disables clipping. The internal count is Clip Limit x pixel count "
            "/ Unique Levels. The loaded image sets the maximum from PeakRatio."
        )
        self.tone_clahe_clip = FocusWheelDoubleSpinBox()
        self.tone_clahe_clip.setRange(0.0, 100.0)
        self.tone_clahe_clip.setDecimals(1)
        self.tone_clahe_clip.setSingleStep(0.1)
        self.tone_clahe_clip.setValue(2.0)
        self.tone_clahe_clip.setToolTip(
            "0 disables histogram clipping. Each tile uses Clip Limit x tile "
            "pixel count / that tile's Unique Levels."
        )
        self.tone_clahe_grid = FocusWheelSpinBox()
        self.tone_clahe_grid.setRange(1, 64)
        self.tone_clahe_grid.setValue(8)
        self.tone_fieldscale_grid = FocusWheelComboBox()
        for grid_count in (1, 2, 4, 8, 16, 32):
            self.tone_fieldscale_grid.addItem(
                f"{grid_count} x {grid_count}", grid_count
            )
        self.tone_fieldscale_grid.setCurrentIndex(
            self.tone_fieldscale_grid.findData(8)
        )
        self.tone_fieldscale_les_distance = FocusWheelSpinBox()
        self.tone_fieldscale_les_distance.setRange(1, 7)
        self.tone_fieldscale_les_distance.setValue(2)
        self.tone_fieldscale_les_threshold = FocusWheelSpinBox()
        self.tone_fieldscale_les_threshold.setRange(0, FIELDSCALE_RAW14_MAX)
        self.tone_fieldscale_les_threshold.setValue(400)
        self.tone_fieldscale_les_threshold.setSuffix(" RAW14 DN")
        self.tone_fieldscale_message_distance = FocusWheelSpinBox()
        self.tone_fieldscale_message_distance.setRange(1, 7)
        self.tone_fieldscale_message_distance.setValue(1)
        self.tone_fieldscale_message_iterations = FocusWheelSpinBox()
        self.tone_fieldscale_message_iterations.setRange(0, 32)
        self.tone_fieldscale_message_iterations.setValue(7)
        self.tone_fieldscale_max_local_gain = FocusWheelComboBox()
        self.tone_fieldscale_max_local_gain.addItem("Off (Reference)", 0.0)
        for gain in (1, 2, 4, 8, 16, 32):
            self.tone_fieldscale_max_local_gain.addItem(f"{gain} x", float(gain))
        self.tone_fieldscale_max_local_gain.setToolTip(
            "Limits flat-region noise amplification. Off preserves the "
            "public-reference Fieldscale output."
        )
        self.tone_mapping_calculated = QLabel(
            "Load a RAW or image file to show the calculated mapping range."
        )
        self.tone_mapping_calculated.setWordWrap(True)
        self.tone_mapping_calculated.setStyleSheet(
            "color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; "
            "border-radius:5px; padding:7px;"
        )
        self.tone_mapping_reset = QPushButton("Reset Mapping Parameters")
        self.tone_mapping_parameter_widgets = {
            "black_dn": self.tone_black_dn,
            "white_dn": self.tone_white_dn,
            "low_percentile": self.tone_low_percentile,
            "high_percentile": self.tone_high_percentile,
            "percentile_gain_limit": self.tone_percentile_gain_limit,
            "histogram_weighted_grid": self.tone_histogram_weighted_grid,
            "histogram_weighted_clip": self.tone_histogram_weighted_clip,
            "histogram_weighted_redistribute": self.tone_histogram_weighted_redistribute,
            "histogram_weighted_bright_weight": self.tone_histogram_weighted_bright_weight,
            "histogram_weighted_bright_percentile": self.tone_histogram_weighted_bright_percentile,
            "histogram_weighted_gain_limit": self.tone_histogram_weighted_gain_limit,
            "clahe_clip": self.tone_clahe_clip,
            "clahe_grid": self.tone_clahe_grid,
            "grid_count": self.tone_fieldscale_grid,
            "les_distance": self.tone_fieldscale_les_distance,
            "les_threshold_raw14": self.tone_fieldscale_les_threshold,
            "message_passing_distance": self.tone_fieldscale_message_distance,
            "message_passing_iterations": self.tone_fieldscale_message_iterations,
            "max_local_gain": self.tone_fieldscale_max_local_gain,
        }
        self.tone_mapping_parameter_rows = {}
        for label, key in (
            ("Black DN", "black_dn"),
            ("White DN", "white_dn"),
            ("Low Percentile", "low_percentile"),
            ("High Percentile", "high_percentile"),
            ("Gain Limit (Gray / DN)", "percentile_gain_limit"),
            ("Weighted Mapping Grid Size", "histogram_weighted_grid"),
            ("Histogram-weighted Clip Limit (Unique Levels)", "histogram_weighted_clip"),
            ("Clip Redistribution", "histogram_weighted_redistribute"),
            ("Bright Threshold Percentile", "histogram_weighted_bright_percentile"),
            ("Bright Weight", "histogram_weighted_bright_weight"),
            ("Gain Limit (Gray / DN)", "histogram_weighted_gain_limit"),
            ("CLAHE Clip Limit (Tile Unique Levels)", "clahe_clip"),
            ("CLAHE Grid Size", "clahe_grid"),
            ("Grid Count", "grid_count"),
            ("LES Distance", "les_distance"),
            ("LES Threshold", "les_threshold_raw14"),
            ("Message Passing Distance", "message_passing_distance"),
            ("Message Passing Iterations", "message_passing_iterations"),
            ("Max Local Gain", "max_local_gain"),
        ):
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(QLabel(label))
            row_layout.addWidget(self.tone_mapping_parameter_widgets[key])
            self.tone_mapping_parameter_layout.addWidget(row)
            self.tone_mapping_parameter_rows[key] = row
        self.tone_mapping_parameter_layout.addWidget(self.tone_mapping_calculated)
        self.tone_mapping_parameter_layout.addWidget(self.tone_mapping_reset)
        self.raw_pipeline_editor = AlgorithmPipelineEditor(
            allow_step_algorithm_change=False,
            title_text="8-bit Enhancement Pipeline (top to bottom)",
            title_pixel_size=19,
        )
        self.raw_left_preview_stage_combo = FocusWheelComboBox()
        self.raw_preview_stage_combo = FocusWheelComboBox()
        for combo in (
            self.raw_left_preview_stage_combo,
            self.raw_preview_stage_combo,
        ):
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        f.addRow("RAW / Image File", self.raw_algorithm_file)
        f.addRow("RAW to 8-bit Tone Mapping", self.raw_input_mapping_combo)
        f.addRow(self.raw_input_mapping_note)
        f.addRow(self.tone_mapping_parameter_group)
        comparison_section = QWidget()
        comparison_layout = QVBoxLayout(comparison_section)
        comparison_layout.setSizeConstraint(QLayout.SetNoConstraint)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.addWidget(self.raw_pipeline_editor)
        f.addRow(comparison_section)
        f = QFormLayout()
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        comparison_layout.addLayout(f)
        self.raw_comparison_section = comparison_section
        self.algorithm_compare_mode = FocusWheelComboBox()
        self.algorithm_compare_mode.addItems(["Side by Side", "Swipe Compare"])
        for label, combo in (
            ("Left Preview", self.raw_left_preview_stage_combo),
            ("Right Preview", self.raw_preview_stage_combo),
            ("Comparison View", self.algorithm_compare_mode),
        ):
            # Match the Image A/B Name fields: expose the container's layout
            # size hints to QFormLayout, including the styled combo height.
            field = QWidget()
            layout = QVBoxLayout(field)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(combo)
            f.addRow(label, field)
        self.algorithm_name_fields = []
        for label in ("Image A Name", "Image B Name"):
            combo = FocusWheelComboBox()
            combo.addItem("Automatic", "auto")
            for name, tag in DUAL_UVC_IMAGE_NAME_OPTIONS:
                combo.addItem(name, tag)
            combo.addItem("Custom...", "custom")
            custom = QLineEdit()
            custom.setMaxLength(80)
            custom.setPlaceholderText("Enter a name (blank uses automatic name)")
            custom.hide()
            field = QWidget()
            layout = QVBoxLayout(field)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(combo)
            layout.addWidget(custom)
            f.addRow(label, field)
            self.algorithm_name_fields.append((combo, custom))
            combo.currentIndexChanged.connect(self.algorithm_image_name_changed)
            custom.textChanged.connect(self.algorithm_image_name_changed)
        self.algorithm_compare_mode.currentIndexChanged.connect(self.update_dual_compare_mode)

        (
            self.raw_algorithm_quality_group,
            self.raw_algorithm_quality_labels,
        ) = self.create_quality_score_group(
            "Image Quality: Original (A) vs Final Pipeline (B)"
        )
        f.addRow(self.raw_algorithm_quality_group)
        self.create_raw_algorithm_statistics(f)
        self.algorithm_image_name_changed()
        self.stack.addWidget(p)
        self.raw_algorithm_file.edit.textChanged.connect(
            self.invalidate_raw_pipeline_source
        )
        self.raw_algorithm_file.path_selected.connect(
            self.auto_preview_raw_algorithm
        )
        self.raw_algorithm_file.edit.returnPressed.connect(
            self.auto_preview_raw_algorithm
        )
        self.raw_input_mapping_combo.currentIndexChanged.connect(
            self.update_raw_pipeline_ui
        )
        for widget in (
            self.tone_black_dn,
            self.tone_white_dn,
            self.tone_low_percentile,
            self.tone_high_percentile,
            self.tone_percentile_gain_limit,
            self.tone_histogram_weighted_grid,
            self.tone_histogram_weighted_clip,
            self.tone_histogram_weighted_bright_percentile,
            self.tone_clahe_clip,
            self.tone_clahe_grid,
            self.tone_fieldscale_les_distance,
            self.tone_fieldscale_les_threshold,
            self.tone_fieldscale_message_distance,
            self.tone_fieldscale_message_iterations,
        ):
            widget.valueChanged.connect(self.update_raw_pipeline_ui)
        self.tone_histogram_weighted_redistribute.toggled.connect(self.update_raw_pipeline_ui)
        self.tone_histogram_weighted_bright_weight.currentIndexChanged.connect(
            self.update_raw_pipeline_ui
        )
        self.tone_histogram_weighted_gain_limit.valueChanged.connect(
            self.update_raw_pipeline_ui
        )
        self.tone_fieldscale_grid.currentIndexChanged.connect(
            self.update_raw_pipeline_ui
        )
        self.tone_fieldscale_max_local_gain.currentIndexChanged.connect(
            self.update_raw_pipeline_ui
        )
        self.tone_mapping_reset.clicked.connect(self.reset_tone_mapping_parameters)
        self.raw_pipeline_editor.pipeline_changed.connect(
            self.update_raw_pipeline_ui
        )
        self.raw_preview_stage_combo.currentIndexChanged.connect(
            self.update_raw_stage_preview
        )
        self.raw_left_preview_stage_combo.currentIndexChanged.connect(
            self.update_raw_stage_preview
        )
        self.resolution_combo.currentIndexChanged.connect(
            self.refresh_raw_algorithm_source_settings
        )
        self.bit_depth_combo.currentIndexChanged.connect(
            self.refresh_raw_algorithm_source_settings
        )
        self.update_raw_pipeline_ui()

    def invalidate_raw_pipeline_source(self, _value=None):
        self.clear_raw_algorithm_statistics()
        self.raw_algorithm_source = None
        self.raw_pipeline_input = None
        self.raw_pipeline_input_max = 255
        self.raw_pipeline_final = None
        self.raw_pipeline_records = []
        self.raw_algorithm_quality_original = None
        self.raw_algorithm_quality_final = None
        if hasattr(self, "raw_algorithm_quality_labels"):
            self.update_quality_score_labels(
                self.raw_algorithm_quality_labels, None, None
            )
        if self.current_function_index() == len(self.FUNCTIONS) - 2:
            self.current_original = None
            self.current_result = None
            self.preview_original.set_array(None)
            self.preview_result.set_array(None)
            self.run_btn.setEnabled(False)

    def refresh_raw_algorithm_source_settings(self, _value=None):
        self.invalidate_raw_pipeline_source()
        if (
            self.current_function_index()
            == self.FUNCTIONS.index("Image Algorithm Compare")
            and self.raw_algorithm_file.text()
        ):
            self.auto_preview_raw_algorithm()

    def configure_tone_mapping_input_range(self, info, source=None):
        value_max = max(1, int(info["value_max"]))
        previous_max = getattr(self, "_tone_mapping_input_max", None)
        previous_white = self.tone_white_dn.value()
        for widget in (self.tone_black_dn, self.tone_white_dn):
            widget.blockSignals(True)
        self.tone_black_dn.setRange(0, max(0, value_max - 1))
        self.tone_white_dn.setRange(1, value_max)
        self.tone_black_dn.setValue(min(self.tone_black_dn.value(), value_max - 1))
        if previous_max is None or previous_white == previous_max or previous_white > value_max:
            self.tone_white_dn.setValue(value_max)
        for widget in (self.tone_black_dn, self.tone_white_dn):
            widget.setSuffix(" DN")
            widget.blockSignals(False)
        self._tone_mapping_input_max = value_max
        if source is not None:
            source_values = np.clip(source, 0, value_max).astype(
                np.int64, copy=False
            )
            source_histogram = np.bincount(
                source_values.reshape(-1), minlength=value_max + 1
            )
            histogram_peak = int(source_histogram.max())
            unique_levels = int(np.count_nonzero(source_histogram))
            histogram_average = source_values.size / unique_levels
            peak_ratio = histogram_peak / histogram_average
            grid = self.tone_histogram_weighted_grid.value()
            if grid > 1:
                peak_ratio = 0.0
                for _, _, tile in weighted_tiles(source_values, grid):
                    histogram = np.bincount(tile.ravel(), minlength=value_max + 1)
                    tile_ratio = histogram.max() * np.count_nonzero(histogram) / histogram.sum()
                    peak_ratio = max(peak_ratio, float(tile_ratio))
            maximum_clip_limit = max(
                2.0, float(np.ceil(peak_ratio * 10.0) / 10.0)
            )
            maximum_clip_count = max(
                1,
                int(round(
                    maximum_clip_limit * source_values.size / unique_levels
                )),
            )
            self.tone_histogram_weighted_clip.blockSignals(True)
            self.tone_histogram_weighted_clip.setMaximum(maximum_clip_limit)
            self.tone_histogram_weighted_clip.blockSignals(False)
            self.tone_histogram_weighted_clip.setToolTip(
                "0 disables clipping. Internal count = Clip Limit x pixel count "
                f"/ Unique Levels. Unique Levels={unique_levels:,}; "
                f"PeakRatio={peak_ratio:.3f}; maximum Clip={maximum_clip_limit:.1f} "
                f"({maximum_clip_count:,} px/bin; peak={histogram_peak:,} px)."
            )
            if grid > 1:
                self.tone_histogram_weighted_clip.setToolTip(
                    "每格獨立計算：Clip × 該格像素數 / 該格 Unique Levels。0 關閉 clipping。"
                    f"所有格子的最大 PeakRatio={peak_ratio:.3f}；最大 Clip={maximum_clip_limit:.1f}。"
                )

    def sync_image_input_mapping(self):
        from thermal_toolkit.algorithms import weighted_aci
        automatic_clip = weighted_aci.ACTIVE_WEIGHTED_ACI['enabled']
        self.tone_histogram_weighted_clip.setEnabled(not automatic_clip)
        method = self.raw_input_mapping_combo.currentData() or "linear"
        visible_keys = {
            "linear": {"black_dn", "white_dn"},
            "percentile": {"low_percentile", "high_percentile", "percentile_gain_limit"},
            "histogram_weighted": {"histogram_weighted_grid", "histogram_weighted_clip", "histogram_weighted_redistribute", "histogram_weighted_bright_weight", "histogram_weighted_bright_percentile", "histogram_weighted_gain_limit"},
            "clahe": {"clahe_clip", "clahe_grid"},
            "fieldscale": {
                "grid_count", "les_distance", "les_threshold_raw14",
                "message_passing_distance", "message_passing_iterations",
                "max_local_gain",
            },
        }.get(method, set())
        for key, row in self.tone_mapping_parameter_rows.items():
            self.tone_mapping_parameter_group.set_body_widget_visible(row, key in visible_keys)
        fieldscale_local_controls_enabled = (
            self.tone_fieldscale_grid.currentData() != 1
        )
        for widget in (
            self.tone_fieldscale_les_distance,
            self.tone_fieldscale_les_threshold,
            self.tone_fieldscale_message_distance,
            self.tone_fieldscale_message_iterations,
        ):
            widget.setEnabled(fieldscale_local_controls_enabled)
        self.tone_histogram_weighted_redistribute.setEnabled(
            automatic_clip or self.tone_histogram_weighted_clip.value() > 0
        )
        self.tone_mapping_parameter_rows["histogram_weighted_bright_percentile"].setEnabled(
            self.tone_histogram_weighted_bright_weight.currentData() != 1.0
        )
        descriptions = {
            "linear": "Original values between Black DN and White DN are mapped linearly to 0..255.",
            "percentile": "Percentiles are calculated from original values. Gain Limit optionally caps the stretch per native DN, preserving lower-median brightness without rescaling to the full range; fractional limits apply before 8-bit rounding.",
            "histogram_weighted": "Grid 1 uses the global mapping; larger grids use per-tile histograms, clipping and bright percentiles with bilinear LUT interpolation. Edge pixels pad incomplete tiles. Adjacent native-DN histogram bins are weighted by W(i)=H(i)+H(i+1). Clip Limit 0 keeps the original histogram; positive values use pixel count / Unique Levels and the maximum is at least 2, or the largest tile PeakRatio rounded up to 0.1 if higher. Redistribution fills all input DN levels, including empty levels, below the cap. Unallocated excess is discarded. Intervals at or above the selected original input percentile receive the selected Bright Weight, then all weights are normalized by their actual sum.",
            "clahe": "CLAHE calculates each tile independently. Clip Limit 0 disables histogram clipping; positive values use Clip Limit x tile pixel count / that tile's Unique Levels, then redistribute excess across all bins before tile-LUT interpolation.",
            "fieldscale": "Fieldscale receives original values and produces the 8-bit pipeline input.",
        }
        if automatic_clip:
            descriptions['histogram_weighted'] += " Automatic Clip is enabled in Setting > Weighted mapping; the manual Clip value is overridden."
        self.raw_input_mapping_note.setText(
            descriptions.get(method, "")
            + (
                " Gain Limit caps each final LUT's adjacent-DN step after weighting, "
                "preserving its lower-median DN brightness without rescaling."
                if method == "histogram_weighted" else ""
            )
            + " All Algorithm Pipeline steps below receive 8-bit input."
        )
        return method

    def current_tone_mapping_parameters(self):
        values = {
            "black_dn": self.tone_black_dn.value(),
            "white_dn": self.tone_white_dn.value(),
            "low_percentile": self.tone_low_percentile.value(),
            "high_percentile": self.tone_high_percentile.value(),
            "percentile_gain_limit": self.tone_percentile_gain_limit.value(),
            "histogram_weighted_grid": self.tone_histogram_weighted_grid.value(),
            "histogram_weighted_clip": self.tone_histogram_weighted_clip.value(),
            "histogram_weighted_redistribute": self.tone_histogram_weighted_redistribute.isChecked(),
            "histogram_weighted_bright_weight": self.tone_histogram_weighted_bright_weight.currentData(),
            "histogram_weighted_bright_percentile": self.tone_histogram_weighted_bright_percentile.value(),
            "histogram_weighted_gain_limit": self.tone_histogram_weighted_gain_limit.value(),
            "clahe_clip": self.tone_clahe_clip.value(),
            "clahe_grid": self.tone_clahe_grid.value(),
            "grid_count": self.tone_fieldscale_grid.currentData(),
            "les_distance": self.tone_fieldscale_les_distance.value(),
            "les_threshold_raw14": self.tone_fieldscale_les_threshold.value(),
            "message_passing_distance": self.tone_fieldscale_message_distance.value(),
            "message_passing_iterations": self.tone_fieldscale_message_iterations.value(),
            "max_local_gain": float(
                self.tone_fieldscale_max_local_gain.currentData()
            ),
        }
        method = self.raw_input_mapping_combo.currentData() or "linear"
        return {
            key: values[key]
            for key in DEFAULT_TONE_MAPPING_PARAMETERS[method]
        }

    def reset_tone_mapping_parameters(self):
        method = self.raw_input_mapping_combo.currentData() or "linear"
        input_max = getattr(self, "raw_algorithm_info", {}).get(
            "value_max", (1 << self.bit_depth) - 1
        )
        defaults = tone_mapping_parameters(method, input_max=input_max)
        widgets = self.tone_mapping_parameter_widgets
        for key, value in defaults.items():
            widget = widgets[key]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            else:
                widget.setValue(value)
        self.update_raw_pipeline_ui()

    def prepare_raw_pipeline_input(self, raw):
        method = self.sync_image_input_mapping()
        self.update_image_preview_titles()
        info = self.raw_algorithm_info
        mapped, details, calculated, used_parameters = apply_tone_mapping(
            raw,
            method,
            info["value_max"],
            self.current_tone_mapping_parameters(),
            pad_fieldscale_input=info["source_type"] == "image",
        )
        self.raw_tone_mapping_calculated = calculated
        self.raw_tone_mapping_parameters = used_parameters
        calculated_text = (
            f"Input: {info['bit_depth']}-bit {info['value_name']} 0..{info['value_max']}\n"
            f"Used: {details}\nOutput: 8-bit 0..255"
        )
        if method in ("histogram_weighted", "percentile"):
            calculated_text += f"; actual range {calculated['output_min']}..{calculated['output_max']}"
        if method == "histogram_weighted":
            calculated_text = weighted_mapping_details_html(info, calculated, used_parameters)
        self.tone_mapping_calculated.setText(calculated_text)
        self.tone_mapping_calculated.setStyleSheet(
            "color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; "
            "border-radius:5px; padding:7px;"
        )
        return mapped, 255, details

    def update_raw_pipeline_ui(self, _value=None):
        if self.raw_algorithm_source is not None:
            self.configure_tone_mapping_input_range(self.raw_algorithm_info, self.raw_algorithm_source)
        self.sync_image_input_mapping()
        self.raw_algorithm_quality_original = None
        self.raw_algorithm_quality_final = None
        if hasattr(self, "raw_algorithm_quality_labels"):
            self.update_quality_score_labels(
                self.raw_algorithm_quality_labels, None, None
            )
        self.rebuild_pipeline_preview_combo(
            self.raw_pipeline_editor,
            self.raw_left_preview_stage_combo,
            include_original=True,
        )
        self.rebuild_pipeline_preview_combo(
            self.raw_pipeline_editor, self.raw_preview_stage_combo
        )
        if self.current_function_index() == len(self.FUNCTIONS) - 2:
            self.run_btn.setEnabled(
                self.raw_algorithm_source is not None
            )
        self.refresh_raw_pipeline_preview()
        self.update_raw_stage_preview()

    def refresh_raw_pipeline_preview(self):
        if self.raw_algorithm_source is None:
            return False
        try:
            pipeline_input, input_max, _mapping_details = (
                self.prepare_raw_pipeline_input(self.raw_algorithm_source)
            )
            final_image, records = apply_algorithm_pipeline(
                pipeline_input,
                self.raw_pipeline_editor.stages(),
                input_max=input_max, allow_empty=True,
                pad_fieldscale_input=self.raw_algorithm_info["source_type"] == "image",
            )
            quality_original = image_quality_analysis(
                self.raw_algorithm_source, self.raw_algorithm_info["value_max"],
                reference=self.raw_algorithm_source, reference_max=self.raw_algorithm_info["value_max"]
            )
            quality_final = image_quality_analysis(final_image, 255, reference=self.raw_algorithm_source, reference_max=self.raw_algorithm_info["value_max"])
        except Exception as exc:
            self.raw_pipeline_input = None
            self.raw_pipeline_input_max = 255
            self.raw_pipeline_final = None
            self.raw_pipeline_records = []
            self.update_raw_algorithm_statistics(self.raw_algorithm_source)
            self.current_result = None
            if self.current_function_index() == len(self.FUNCTIONS) - 2:
                self.preview_result.set_array(None)
                self.run_btn.setEnabled(False)
            self.tone_mapping_calculated.setText(f"Invalid mapping: {exc}")
            self.tone_mapping_calculated.setStyleSheet(
                "color:#ff9d9d; background:#35292b; border-left:3px solid #e36b6b; "
                "border-radius:5px; padding:7px;"
            )
            self.log.append(f"RAW pipeline preview error: {exc}")
            return False
        self.raw_pipeline_input = pipeline_input
        self.raw_pipeline_input_max = input_max
        self.raw_pipeline_final = final_image
        self.raw_pipeline_records = records
        self.update_raw_algorithm_statistics(self.raw_algorithm_source)
        self.raw_algorithm_quality_original = quality_original
        self.raw_algorithm_quality_final = quality_final
        self.update_quality_score_labels(
            self.raw_algorithm_quality_labels,
            quality_original,
            quality_final,
        )
        self.current_original = raw_to_display_u8(
            self.raw_algorithm_source, self.raw_algorithm_info["value_max"]
        )
        if self.current_function_index() == len(self.FUNCTIONS) - 2:
            self.preview_original.set_array(self.current_original)
            self.run_btn.setEnabled(
                self.raw_algorithm_source is not None
            )
        self.update_raw_stage_preview()
        return True

    def update_raw_stage_preview(self, _combo_index=None):
        self.algorithm_image_name_changed()
        if self.raw_pipeline_final is None:
            return
        left_preview = self.raw_pipeline_preview_image(
            self.raw_left_preview_stage_combo.currentData()
        )
        right_preview = self.raw_pipeline_preview_image(
            self.raw_preview_stage_combo.currentData()
        )
        self.current_result = right_preview
        if self.current_function_index() == len(self.FUNCTIONS) - 2:
            left_title, right_title = self.algorithm_image_names()
            self.preview_original.set_title(left_title)
            self.preview_original_caption.setText(left_title)
            self.preview_original.set_array(left_preview)
            self.preview_result.set_title(right_title)
            self.preview_result_caption.setText(right_title)
            self.preview_result.set_array(right_preview)

    def raw_pipeline_preview_image(self, selected_pipeline_index):
        if selected_pipeline_index == "original":
            return self.current_original
        if selected_pipeline_index == 0:
            return self.raw_pipeline_input
        return pipeline_record_image(
            self.raw_pipeline_records,
            selected_pipeline_index,
            self.raw_pipeline_final,
        )

    def algorithm_image_names(self):
        kind = "RAW" if getattr(self, "raw_algorithm_info", {}).get("source_type") == "raw" else "Image"
        defaults = (
            self.raw_left_preview_stage_combo.currentText()
            or f"Original {kind} (Fixed Scale)",
            self.raw_preview_stage_combo.currentText() or "Final Pipeline",
        )
        return tuple(
            (custom.text().strip() or default) if combo.currentData() == "custom"
            else default if combo.currentData() == "auto" else combo.currentText()
            for (combo, custom), default in zip(self.algorithm_name_fields, defaults)
        )

    def algorithm_image_name_changed(self, _value=None):
        for combo, custom in self.algorithm_name_fields:
            custom.setVisible(combo.currentData() == "custom")
        names = self.algorithm_image_names()
        quality_headers = getattr(
            getattr(self, "raw_algorithm_quality_group", None),
            "comparison_header_labels",
            (),
        )
        if len(quality_headers) == 2:
            quality_headers[0].setText(names[0])
            quality_headers[1].setText(names[1])
        if self.FUNCTIONS[self.current_function_index()] == "Image Algorithm Compare":
            self.update_image_preview_titles()
            self.update_image_preview_window(self.preview_original)

    def auto_preview_raw_algorithm(self, _path=None):
        if (
            self.current_function_index()
            != self.FUNCTIONS.index("Image Algorithm Compare")
            or not self.raw_algorithm_file.text()
        ):
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            raw_path = self.raw_algorithm_file.text()
            self.raw_algorithm_source, self.raw_algorithm_info = self.load_analysis_source(raw_path)
            self.configure_tone_mapping_input_range(
                self.raw_algorithm_info, self.raw_algorithm_source
            )
            self.update_raw_algorithm_statistics(self.raw_algorithm_source)
            if not self.refresh_raw_pipeline_preview():
                return
            self.log.append(f"\n[{self.raw_algorithm_info['source_type'].title()}] Preview loaded: {raw_path}")
            self.log.append(
                f"Pipeline preview: {self.raw_pipeline_editor.summary()}"
            )
            self.log.append("Image Quality A: " + quality_text(self.raw_algorithm_quality_original))
            self.log.append("Image Quality B: " + quality_text(self.raw_algorithm_quality_final))
            self.log.append(
                "Preview updated. Save Analysis Results creates output files."
            )
        except Exception as exc:
            self.invalidate_raw_pipeline_source()
            self.log.append(f"\nERROR: {exc}")
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def run_raw_algorithm(self):
        raw, self.raw_algorithm_info = self.load_analysis_source(self.raw_algorithm_file.text())
        self.configure_tone_mapping_input_range(self.raw_algorithm_info, raw)
        self.update_raw_algorithm_statistics(raw)
        raw8 = raw_to_display_u8(raw, self.raw_algorithm_info["value_max"])
        stages = self.raw_pipeline_editor.stages()
        pipeline_input, input_max, mapping_details = (
            self.prepare_raw_pipeline_input(raw)
        )
        total_started = time.perf_counter()
        processed, records = apply_algorithm_pipeline(
            pipeline_input,
            stages,
            input_max=input_max, allow_empty=True,
            pad_fieldscale_input=self.raw_algorithm_info["source_type"] == "image",
        )
        total_ms = (time.perf_counter() - total_started) * 1000.0

        # Both displayed outputs are evaluated against the same native reference.
        quality_original = image_quality_analysis(raw, self.raw_algorithm_info["value_max"], reference=raw, reference_max=self.raw_algorithm_info["value_max"])
        quality_final = image_quality_analysis(processed, 255, reference=raw, reference_max=self.raw_algorithm_info["value_max"])
        raw_stats, raw_histogram = raw_dn_image_analysis(
            raw, self.raw_algorithm_info["value_max"]
        )
        raw_histogram = np.bincount(
            np.clip(raw, 0, self.raw_algorithm_info["value_max"])
            .astype(np.int64, copy=False)
            .reshape(-1),
            minlength=self.raw_algorithm_info["value_max"] + 1,
        ).astype(np.int64)
        image_stats, image_histogram = raw_dn_image_analysis(processed, 255)
        self.raw_algorithm_quality_original = quality_original
        self.raw_algorithm_quality_final = quality_final
        self.update_quality_score_labels(
            self.raw_algorithm_quality_labels,
            quality_original,
            quality_final,
        )

        out = self.make_output("raw_algorithm_compare")
        original_path = out / "reference_linear.png"
        mapping_method = self.raw_input_mapping_combo.currentData() or "linear"
        mapping_path = out / f"tone_mapped_{mapping_method}.png"
        result_path = out / "final_pipeline.png"
        overview_path = out / "00_pipeline_overview.png"
        display_original_path = out / "display_original.png"
        display_final_path = out / "display_final.png"
        quality_excel_path = out / "quality_analysis.xlsx"
        raw_analysis_excel_path = out / "raw_analysis.xlsx"
        image_analysis_excel_path = out / "image_analysis.xlsx"
        save_png(original_path, raw8)
        raw_csv_path = None
        if self.raw_algorithm_info["source_type"] == "raw":
            raw_csv_path = out / "original_raw_dn.csv"
            save_csv_matrix(raw_csv_path, raw, "%d")
            self.log.append(f"Saved original RAW DN CSV: {raw_csv_path}")
        save_png(mapping_path, pipeline_input)
        weighted_intervals_path = None
        if mapping_method == "histogram_weighted":
            weighted_intervals_path = out / "histogram_weighted_intervals.csv"
            save_histogram_weighted_intervals(
                weighted_intervals_path, raw, self.raw_algorithm_info["value_max"],
                self.raw_tone_mapping_parameters,
            )
            self.log.append(f"Saved interval ratios: {weighted_intervals_path}")
        for record in records:
            step_path = out / (
                f"step_{record['pipeline_index']:02d}_"
                f"{record['algorithm']}.png"
            )
            save_png(step_path, record["image"])
            record["output_file"] = step_path.name
        save_png(result_path, processed)
        save_png(display_original_path, quality_original["display_u8"])
        save_png(display_final_path, quality_final["display_u8"])

        mapping_label = self.raw_input_mapping_combo.currentText()
        overview_items = [
            ("00 Linear Reference", raw8, 255),
            (f"01 Tone Mapping - {mapping_label}", pipeline_input, input_max),
        ]
        for overview_index, record in enumerate(records, start=2):
            overview_items.append(
                (
                    f"{overview_index:02d} Step {record['pipeline_index']:02d} "
                    f"{record['name']}",
                    record["image"],
                    255,
                )
            )
        save_pipeline_overview_png(overview_path, overview_items)
        self.raw_algorithm_source = raw
        self.raw_pipeline_input = pipeline_input
        self.raw_pipeline_input_max = input_max
        self.raw_pipeline_final = processed
        self.update_raw_algorithm_statistics(raw)
        self.raw_pipeline_records = records
        self.current_original = raw8
        self.update_image_preview_titles()
        self.update_raw_stage_preview()
        comparison_path = out / "image_comparison.png"
        if self.algorithm_compare_mode.currentIndex() == 1:
            self.save_dual_swipe_preview(comparison_path)
        else:
            comparison = self.compose_dual_image_preview(
                [
                    self.raw_pipeline_preview_image(
                        self.raw_left_preview_stage_combo.currentData()
                    ),
                    self.current_result,
                ],
                self.algorithm_image_names(),
            )
            save_rgb_png(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
        self.log.append(f"Saved comparison preview: {comparison_path}")

        manifest = {
            "format_version": 3,
            "mode": "raw_algorithm_compare",
            "comparison_view": self.algorithm_compare_mode.currentText(),
            "swipe_position": self.dual_swipe_preview.position,
            "comparison_output": comparison_path.name,
            "comparison_image_names": list(self.algorithm_image_names()),
            "comparison_left_stage": self.raw_left_preview_stage_combo.currentText(),
            "comparison_left_stage_index": self.raw_left_preview_stage_combo.currentData(),
            "comparison_stage": self.raw_preview_stage_combo.currentText(),
            "comparison_stage_index": self.raw_preview_stage_combo.currentData(),
            "source": str(Path(self.raw_algorithm_file.text()).resolve()),
            "resolution": {"width": raw.shape[1], "height": raw.shape[0]},
            "source_info": self.raw_algorithm_info,
            "bit_depth": self.raw_algorithm_info["bit_depth"],
            "quality_profile": "Fieldscale IQA",
            "tone_mapping": {
                "method": mapping_method,
                "name": self.raw_input_mapping_combo.currentText(),
                "input_domain": "native_dn",
                "output_domain": "display_u8",
                "parameters": dict(self.raw_tone_mapping_parameters),
                "calculated": dict(self.raw_tone_mapping_calculated),
                "details": mapping_details,
                "input_max": self.raw_algorithm_info["value_max"],
                "output_max": 255,
                "output_file": mapping_path.name,
                "interval_ratios_output": (
                    weighted_intervals_path.name if weighted_intervals_path else None
                ),
            },
            "pipeline": self.raw_pipeline_editor.configuration(),
            "execution": pipeline_execution_json(records),
            "total_ms": round(total_ms, 3),
            "original_output": original_path.name,
            "raw_csv_output": (
                raw_csv_path.name if raw_csv_path is not None else None
            ),
            "final_output": result_path.name,
            "overview_output": overview_path.name,
            "quality_original": quality_analysis_manifest(quality_original),
            "quality_final": quality_analysis_manifest(quality_final),
            "quality_layer_outputs": {
                "display_original": display_original_path.name,
                "display_final": display_final_path.name,
            },
            "quality_excel_output": quality_excel_path.name,
            "raw_analysis_excel_output": raw_analysis_excel_path.name,
            "image_analysis_excel_output": image_analysis_excel_path.name,
        }
        for execution, record in zip(manifest["execution"], records):
            if "output_file" in record:
                execution["output_file"] = record["output_file"]
        analysis_metadata = {
            key: value
            for key, value in manifest.items()
            if key not in {"quality_original", "quality_final"}
        }
        save_histogram_excel(
            raw_analysis_excel_path,
            "Image Algorithm Compare - Original RAW Analysis",
            analysis_metadata,
            [
                {
                    "label": "Original RAW",
                    "stats": raw_stats,
                    "histogram": raw_histogram,
                }
            ],
            value_max=self.raw_algorithm_info["value_max"],
            value_name=self.raw_algorithm_info["value_name"],
        )
        save_histogram_excel(
            image_analysis_excel_path,
            "Image Algorithm Compare - Final Pipeline Analysis",
            analysis_metadata,
            [
                {
                    "label": "Final Pipeline Output",
                    "stats": image_stats,
                    "histogram": image_histogram,
                }
            ],
            value_max=255,
            value_name="8-bit Gray",
        )
        with open(out / "pipeline.json", "w", encoding="utf-8") as destination:
            manifest["weighted_mapping_settings"] = dict(self.app_settings["weighted_mapping"])
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(out / "pipeline_summary.txt", "w", encoding="utf-8") as summary:
            summary.write(f"Source: {manifest['source']}\n")
            summary.write(f"Tone Mapping: {mapping_details}\n")
            summary.write(f"Pipeline: {self.raw_pipeline_editor.summary()}\n")
            summary.write(f"Overview: {overview_path.name}\n")
            summary.write(
                "Quality: Gradient/Entropy and reference-based contrast loss/TMQI; no total score.\n"
            )
            write_quality_summary(summary, "Original RAW", quality_original)
            write_quality_summary(summary, "Final Pipeline", quality_final)
            summary.write(f"Quality Excel: {quality_excel_path.name}\n")
            summary.write(f"Original RAW analysis: {raw_analysis_excel_path.name}\n")
            summary.write(f"Final 8-bit analysis: {image_analysis_excel_path.name}\n")
            for execution in manifest["execution"]:
                summary.write(
                    f"Step {execution['pipeline_index']}: {execution['name']} - "
                    f"{execution['elapsed_ms']:.3f} ms - "
                    f"{' | '.join(execution['details'])}\n"
                )
            summary.write(f"Pipeline Total: {total_ms:.3f} ms\n")

        save_quality_excel(
            quality_excel_path,
            "Image Algorithm Quality Compare",
            {
                key: value
                for key, value in manifest.items()
                if key not in {"quality_original", "quality_final"}
            },
            [
                {
                    "label": "Original RAW",
                    "metrics": quality_original["metrics"],
                    "unavailable_reasons": quality_original.get("unavailable_reasons", {}),
                },
                {
                    "label": "Final Pipeline",
                    "metrics": quality_final["metrics"],
                    "unavailable_reasons": quality_final.get("unavailable_reasons", {}),
                },
            ],
        )

        self.log.append(f"Tone Mapping: {mapping_details}")
        self.log.append(f"Pipeline: {self.raw_pipeline_editor.summary()}")
        for record in records:
            self.log.append(
                f"Step {record['pipeline_index']} {record['name']}: "
                f"{record['elapsed_ms']:.3f} ms | "
                f"{' | '.join(record['details'])}"
            )
        self.log.append(f"Pipeline total: {total_ms:.3f} ms")
        self.log.append("Image Quality A: " + quality_text(quality_original))
        self.log.append("Image Quality B: " + quality_text(quality_final))
        self.log_stats("Original 8-bit", raw8)
        self.log_stats("Final 8-bit", processed)
        self.log.append(f"Saved: {original_path}")
        self.log.append(f"Saved: {result_path}")
        self.log.append(f"Saved overview: {overview_path}")
        self.log.append(f"Saved quality analysis: {quality_excel_path}")
        self.log.append(f"Saved original RAW analysis: {raw_analysis_excel_path}")
        self.log.append(f"Saved final 8-bit analysis: {image_analysis_excel_path}")
        self.log.append(f"Pipeline settings: {out / 'pipeline.json'}")
