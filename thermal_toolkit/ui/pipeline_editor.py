"""ui / pipeline_editor."""

from thermal_toolkit.algorithms.fieldscale import FIELDSCALE_RAW14_MAX
from thermal_toolkit.algorithms.parameters import (
    AlgorithmStage,
    IMAGE_ALGORITHM_DESCRIPTIONS,
    PIPELINE_ALGORITHM_OPTIONS,
    algorithm_display_name,
    default_algorithm_parameters,
)
from thermal_toolkit.algorithms.pipeline import pipeline_summary_text
from thermal_toolkit.core.qt import (
    QCheckBox,
    QComboBox,
    QFont,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSignal,
)
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox, QualityScoreCheckBox
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox, FocusWheelDoubleSpinBox, FocusWheelSpinBox


class AlgorithmPipelineEditor(QWidget):
    pipeline_changed = pyqtSignal()

    def __init__(
        self,
        parent=None,
        allow_step_algorithm_change=True,
        fieldscale_temporal=False,
        title_text="Algorithm Pipeline (top to bottom)",
        title_pixel_size=15,
    ):
        super().__init__(parent)
        self._stages = []
        self._updating = False
        self._allow_step_algorithm_change = bool(
            allow_step_algorithm_change
        )
        self._fieldscale_temporal = bool(fieldscale_temporal)
        self._parameter_size_update_pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)
        self.pipeline_group = CollapsibleGroupBox(title_text)
        self.pipeline_group.setObjectName("pipelineParameterCard")
        title_font = self.pipeline_group.font()
        title_font.setPixelSize(int(title_pixel_size))
        title_font.setWeight(QFont.Bold)
        self.pipeline_group.setFont(title_font)
        outer_root = root
        outer_root.addWidget(self.pipeline_group)
        root = QVBoxLayout(self.pipeline_group)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(7)
        self.pipeline_group.expanded_changed.connect(self.update_parameter_size_constraints)

        self.add_algorithm_combo = FocusWheelComboBox()
        for display_name, algorithm_key in PIPELINE_ALGORITHM_OPTIONS:
            self.add_algorithm_combo.addItem(display_name, algorithm_key)
        self.add_algorithm_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_button = QPushButton("Add Step")
        self.add_button.setObjectName("primaryButton")
        root.addWidget(self.add_algorithm_combo)
        root.addWidget(self.add_button)

        self.stage_list = QListWidget()
        self.stage_list.setMinimumHeight(132)
        self.stage_list.setMaximumHeight(190)
        self.stage_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stage_list.setTextElideMode(Qt.ElideRight)
        self.stage_list.setToolTip(
            "Checked steps run from top to bottom. Duplicate algorithms are allowed."
        )
        root.addWidget(self.stage_list)

        order_grid = QGridLayout()
        order_grid.setHorizontalSpacing(7)
        order_grid.setVerticalSpacing(7)
        self.move_up_button = QPushButton("Move Up")
        self.move_down_button = QPushButton("Move Down")
        self.remove_button = QPushButton("Remove")
        self.reset_all_parameters_button = QPushButton("Reset All Parameters")
        self.reset_all_parameters_button.setToolTip(
            "Reset parameters for all listed steps, including unchecked steps."
        )
        self.reset_pipeline_button = QPushButton("Reset Pipeline")
        self.reset_pipeline_button.setObjectName("resetButton")
        order_grid.addWidget(self.move_up_button, 0, 0)
        order_grid.addWidget(self.move_down_button, 0, 1)
        order_grid.addWidget(self.remove_button, 1, 0)
        order_grid.addWidget(self.reset_all_parameters_button, 1, 1)
        order_grid.addWidget(self.reset_pipeline_button, 2, 0, 1, 2)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.remove_button,
            self.reset_all_parameters_button,
            self.reset_pipeline_button,
        ):
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        root.addLayout(order_grid)

        self.parameter_group = CollapsibleGroupBox("Step Parameters")
        self.parameter_group.setObjectName("pipelineParameterCard")
        self.parameter_group.expanded_changed.connect(
            self.update_parameter_size_constraints
        )
        self.parameter_form = QVBoxLayout(self.parameter_group)
        self.parameter_form.setContentsMargins(10, 10, 10, 10)
        self.parameter_form.setSpacing(8)
        self.reset_step_button = QPushButton("Reset Step")
        self.reset_step_button.setToolTip(
            "Reset parameters for the currently selected step."
        )
        self.parameter_rows = {}
        self.stage_algorithm_combo = FocusWheelComboBox()
        for display_name, algorithm_key in PIPELINE_ALGORITHM_OPTIONS:
            self.stage_algorithm_combo.addItem(display_name, algorithm_key)
        self.stage_algorithm_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.stage_description = QLabel()
        self.stage_description.setWordWrap(True)
        self.stage_description.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.stage_description.setStyleSheet(
            "color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; "
            "border-radius:5px; padding:7px;"
        )
        self.stage_low_percentile = FocusWheelDoubleSpinBox()
        self.stage_low_percentile.setRange(0.0, 99.5)
        self.stage_low_percentile.setDecimals(1)
        self.stage_low_percentile.setSingleStep(0.5)
        self.stage_high_percentile = FocusWheelDoubleSpinBox()
        self.stage_high_percentile.setRange(0.5, 100.0)
        self.stage_high_percentile.setDecimals(1)
        self.stage_high_percentile.setSingleStep(0.5)
        self.stage_percentile_gain_limit = FocusWheelDoubleSpinBox()
        self.stage_percentile_gain_limit.setRange(0.0, 4.0)
        self.stage_percentile_gain_limit.setDecimals(1)
        self.stage_percentile_gain_limit.setSingleStep(0.1)
        self.stage_percentile_gain_limit.setSpecialValueText("0.0 (Off)")
        self.stage_percentile_gain_limit.setToolTip(
            "Cap percentile stretching per 8-bit input level, not original RAW DN. "
            "0.0 keeps the original mapping. Preserve lower-median brightness without rescaling. "
            "Fractional limits apply before output rounding; later steps may increase contrast."
        )
        self.stage_clahe_clip = FocusWheelDoubleSpinBox()
        self.stage_clahe_clip.setRange(0.1, 100.0)
        self.stage_clahe_clip.setDecimals(1)
        self.stage_clahe_clip.setSingleStep(0.1)
        self.stage_clahe_grid = FocusWheelSpinBox()
        self.stage_clahe_grid.setRange(1, 64)
        self.stage_histogram_weighted_grid = FocusWheelSpinBox()
        self.stage_histogram_weighted_grid.setRange(1, 64)
        self.stage_histogram_weighted_grid.setToolTip("Tiles per axis: 1 is global; larger grids use bilinear LUT interpolation.")
        self.stage_histogram_weighted_clip = FocusWheelDoubleSpinBox()
        self.stage_histogram_weighted_clip.setRange(0.0, 256.0)
        self.stage_histogram_weighted_clip.setDecimals(1)
        self.stage_histogram_weighted_clip.setSingleStep(0.1)
        self.stage_histogram_weighted_clip.setToolTip(
            "0 disables clipping. Threshold = Clip Limit x tile pixels / populated 8-bit levels."
        )
        self.stage_histogram_weighted_redistribute = QualityScoreCheckBox("Redistribute")
        self.stage_histogram_weighted_redistribute.setMinimumWidth(0)
        self.stage_histogram_weighted_redistribute.setToolTip(
            "Redistribute across all input levels, including empty levels, below the cap; discard excess that cannot fit."
        )
        self.stage_histogram_weighted_bright_percentile = FocusWheelDoubleSpinBox()
        self.stage_histogram_weighted_bright_percentile.setRange(0.0, 100.0)
        self.stage_histogram_weighted_bright_percentile.setDecimals(1)
        self.stage_histogram_weighted_bright_percentile.setSingleStep(1.0)
        self.stage_histogram_weighted_bright_percentile.setToolTip("Percentile of this step's input, before clipping; calculated per tile.")
        self.stage_histogram_weighted_bright_weight = FocusWheelComboBox()
        for weight in (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
            self.stage_histogram_weighted_bright_weight.addItem("1 (Off)" if weight == 1 else f"{weight:g}", weight)
        self.stage_histogram_weighted_gain_limit = FocusWheelDoubleSpinBox()
        self.stage_histogram_weighted_gain_limit.setRange(0.0, 4.0)
        self.stage_histogram_weighted_gain_limit.setDecimals(1)
        self.stage_histogram_weighted_gain_limit.setSingleStep(0.1)
        self.stage_histogram_weighted_gain_limit.setSpecialValueText("0.0 (Off)")
        self.stage_histogram_weighted_gain_limit.setToolTip(
            "Maximum LUT increase per 8-bit input level (not original RAW DN). "
            "Keep lower-median brightness without rescaling; applied per tile before interpolation."
            " Enter 0.0..4.0 in steps of 0.1. Fractions accumulate before rounding; "
            "8-bit output steps may exceed the fractional limit due to rounding."
        )
        self.histogram_weighted_widgets = {
            key: getattr(self, f"stage_{key}")
            for key in default_algorithm_parameters("histogram_weighted")
        }
        self.stage_gamma = FocusWheelDoubleSpinBox()
        self.stage_gamma.setRange(0.1, 3.0)
        self.stage_gamma.setDecimals(1)
        self.stage_gamma.setSingleStep(0.1)
        self.stage_median_kernel_size = FocusWheelComboBox()
        self.stage_gaussian_kernel_size = FocusWheelComboBox()
        for kernel_size in (3, 5, 7, 9, 11, 13, 15):
            label = f"{kernel_size} x {kernel_size}"
            self.stage_median_kernel_size.addItem(label, kernel_size)
            self.stage_gaussian_kernel_size.addItem(label, kernel_size)
        self.stage_gaussian_kernel_size.setCurrentIndex(
            self.stage_gaussian_kernel_size.findData(5)
        )
        self.stage_gaussian_sigma = FocusWheelDoubleSpinBox()
        self.stage_gaussian_sigma.setRange(0.1, 10.0)
        self.stage_gaussian_sigma.setDecimals(1)
        self.stage_gaussian_sigma.setSingleStep(0.1)
        self.stage_gaussian_sigma.setValue(1.0)
        self.stage_bilateral_strength = FocusWheelSpinBox()
        self.stage_bilateral_strength.setRange(1, 7)
        self.stage_unsharp_amount = FocusWheelDoubleSpinBox()
        self.stage_unsharp_amount.setRange(0.1, 3.0)
        self.stage_unsharp_amount.setDecimals(1)
        self.stage_unsharp_amount.setSingleStep(0.1)
        self.stage_dde_sigma_s = FocusWheelSpinBox()
        self.stage_dde_sigma_s.setRange(1, 256)
        self.stage_dde_sigma_s.setValue(21)
        self.stage_dde_sigma_s.setToolTip("Spatial smoothing radius in pixels.")
        self.stage_dde_sigma_r = FocusWheelSpinBox()
        self.stage_dde_sigma_r.setRange(1, 255)
        self.stage_dde_sigma_r.setValue(51)
        self.stage_dde_sigma_r.setToolTip("Edge sensitivity in Y8 DN.")
        self.stage_dde_gain_percent = FocusWheelSpinBox()
        self.stage_dde_gain_percent.setRange(0, 1000)
        self.stage_dde_gain_percent.setValue(200)
        self.stage_dde_gain_percent.setToolTip("Detail gain: 100% preserves input; 0% outputs the base.")
        self.stage_dde_num_iters = FocusWheelSpinBox()
        self.stage_dde_num_iters.setRange(1, 8)
        self.stage_dde_num_iters.setValue(3)
        self.stage_dde_num_iters.setToolTip("Standard: 3 iterations; Fast: 1 iteration.")
        self.stage_fieldscale_grid_count = FocusWheelComboBox()
        for grid_count in (1, 2, 4, 8, 16, 32):
            self.stage_fieldscale_grid_count.addItem(
                f"{grid_count} x {grid_count}", grid_count
            )
        self.stage_fieldscale_grid_count.setCurrentIndex(
            self.stage_fieldscale_grid_count.findData(8)
        )
        self.stage_fieldscale_grid_count.setToolTip(
            "1x1 uses global Min/Max; larger values use local regions. Firmware default is 8x8."
        )
        self.stage_fieldscale_les_distance = FocusWheelSpinBox()
        self.stage_fieldscale_les_distance.setRange(1, 7)
        self.stage_fieldscale_les_distance.setValue(2)
        self.stage_fieldscale_les_distance.setToolTip(
            "Neighborhood radius on the 8x8 Min/Max grid; firmware default is 2."
        )
        self.stage_fieldscale_les_threshold = FocusWheelSpinBox()
        self.stage_fieldscale_les_threshold.setRange(0, FIELDSCALE_RAW14_MAX)
        self.stage_fieldscale_les_threshold.setValue(400)
        self.stage_fieldscale_les_threshold.setSuffix(" RAW14 DN")
        self.stage_fieldscale_les_threshold.setToolTip(
            "Local-extrema suppression threshold in RAW14 DN."
        )
        self.stage_fieldscale_message_distance = FocusWheelSpinBox()
        self.stage_fieldscale_message_distance.setRange(1, 7)
        self.stage_fieldscale_message_distance.setValue(1)
        self.stage_fieldscale_message_distance.setToolTip(
            "Neighborhood radius for each message-passing iteration."
        )
        self.stage_fieldscale_message_iterations = FocusWheelSpinBox()
        self.stage_fieldscale_message_iterations.setRange(0, 32)
        self.stage_fieldscale_message_iterations.setValue(7)
        self.stage_fieldscale_message_iterations.setToolTip(
            "Number of message-passing iterations; 0 disables message passing."
        )
        self.stage_fieldscale_max_local_gain = FocusWheelComboBox()
        self.stage_fieldscale_max_local_gain.addItem("Off (Reference)", 0.0)
        for gain in (1, 2, 4, 8, 16, 32):
            self.stage_fieldscale_max_local_gain.addItem(f"{gain} x", float(gain))
        self.stage_fieldscale_max_local_gain.setToolTip(
            "Limits noise amplification by enforcing local range >= full input "
            "range / selected gain. Off preserves the reference algorithm."
        )
        self.stage_fieldscale_temporal_smoothing = FocusWheelSpinBox()
        self.stage_fieldscale_temporal_smoothing.setRange(0, 99)
        self.stage_fieldscale_temporal_smoothing.setValue(90)
        self.stage_fieldscale_temporal_smoothing.setSuffix(" % previous frame")
        self.stage_fieldscale_temporal_smoothing.setToolTip(
            "UVC only: previous-frame weight. 0% disables temporal smoothing."
        )
        if self._allow_step_algorithm_change:
            self._add_parameter_row("Algorithm", self.stage_algorithm_combo)
        self.parameter_form.addWidget(self.stage_description)
        self._add_parameter_row(
            "Grid 數量", self.stage_fieldscale_grid_count
        )
        self._add_parameter_row(
            "LES Distance", self.stage_fieldscale_les_distance
        )
        self._add_parameter_row(
            "LES Threshold", self.stage_fieldscale_les_threshold
        )
        self._add_parameter_row(
            "Message Passing Distance", self.stage_fieldscale_message_distance
        )
        self._add_parameter_row(
            "Message Passing 次數 N",
            self.stage_fieldscale_message_iterations,
        )
        self._add_parameter_row(
            "Max Local Gain", self.stage_fieldscale_max_local_gain
        )
        self._add_parameter_row(
            "Temporal Smoothing",
            self.stage_fieldscale_temporal_smoothing,
        )
        self._add_parameter_row("Low Percentile", self.stage_low_percentile)
        self._add_parameter_row("High Percentile", self.stage_high_percentile)
        self._add_parameter_row("Gain Limit (Gray / DN)", self.stage_percentile_gain_limit)
        self._add_parameter_row("CLAHE Clip Limit", self.stage_clahe_clip)
        self._add_parameter_row("CLAHE Grid Size", self.stage_clahe_grid)
        for label, key in (
            ("Weighted Mapping Grid Size", "histogram_weighted_grid"),
            ("Clip Limit (Unique Levels)", "histogram_weighted_clip"),
            ("Clip Redistribution", "histogram_weighted_redistribute"),
            ("Bright Threshold Percentile", "histogram_weighted_bright_percentile"),
            ("Bright Weight", "histogram_weighted_bright_weight"),
            ("Gain Limit (Gray / DN)", "histogram_weighted_gain_limit"),
        ):
            self._add_parameter_row(label, self.histogram_weighted_widgets[key])
        self._add_parameter_row("Gamma", self.stage_gamma)
        self._add_parameter_row("Median Kernel Size", self.stage_median_kernel_size)
        self._add_parameter_row("Gaussian Kernel Size", self.stage_gaussian_kernel_size)
        self._add_parameter_row("Gaussian Sigma", self.stage_gaussian_sigma)
        self._add_parameter_row("Sigma S", self.stage_dde_sigma_s)
        self._add_parameter_row("Sigma R", self.stage_dde_sigma_r)
        self._add_parameter_row("Gain (%)", self.stage_dde_gain_percent)
        self._add_parameter_row("Iterations", self.stage_dde_num_iters)
        self._add_parameter_row(
            "Bilateral Strength", self.stage_bilateral_strength
        )
        self._add_parameter_row("Unsharp Amount", self.stage_unsharp_amount)
        self.parameter_form.addWidget(self.reset_step_button)
        self.parameter_form.addStretch()
        outer_root.addWidget(self.parameter_group)

        self.add_button.clicked.connect(self.add_stage)
        self.move_up_button.clicked.connect(lambda: self.move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self.move_selected(1))
        self.remove_button.clicked.connect(self.remove_selected)
        self.reset_step_button.clicked.connect(self.reset_selected_step)
        self.reset_all_parameters_button.clicked.connect(self.reset_all_parameters)
        self.reset_pipeline_button.clicked.connect(self.reset_pipeline)
        self.stage_list.currentRowChanged.connect(self.load_selected_step)
        self.stage_list.itemChanged.connect(self.stage_check_changed)
        if self._allow_step_algorithm_change:
            self.stage_algorithm_combo.currentIndexChanged.connect(
                self.selected_algorithm_changed
            )
        for widget in (
            self.stage_low_percentile,
            self.stage_high_percentile,
            self.stage_percentile_gain_limit,
            self.stage_clahe_clip,
            self.stage_clahe_grid,
            self.stage_gamma,
            self.stage_gaussian_sigma,
            self.stage_bilateral_strength,
            self.stage_unsharp_amount,
            self.stage_dde_sigma_s,
            self.stage_dde_sigma_r,
            self.stage_dde_gain_percent,
            self.stage_dde_num_iters,
            self.stage_fieldscale_les_distance,
            self.stage_fieldscale_les_threshold,
            self.stage_fieldscale_message_distance,
            self.stage_fieldscale_message_iterations,
            self.stage_fieldscale_temporal_smoothing,
        ):
            widget.valueChanged.connect(self.selected_parameter_changed)
        self.stage_fieldscale_grid_count.currentIndexChanged.connect(
            self.selected_fieldscale_grid_changed
        )
        self.stage_median_kernel_size.currentIndexChanged.connect(
            self.selected_parameter_changed
        )
        self.stage_gaussian_kernel_size.currentIndexChanged.connect(
            self.selected_parameter_changed
        )
        self.stage_fieldscale_max_local_gain.currentIndexChanged.connect(
            self.selected_parameter_changed
        )
        self.set_stages([])

        for widget in self.histogram_weighted_widgets.values():
            if isinstance(widget, QCheckBox):
                widget.toggled.connect(self.selected_parameter_changed)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.selected_parameter_changed)
            else:
                widget.valueChanged.connect(self.selected_parameter_changed)

    def _add_parameter_row(self, label_text, widget):
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        row_layout.addWidget(QLabel(label_text))
        row_layout.addWidget(widget)
        self.parameter_form.addWidget(row)
        self.parameter_rows[widget] = row

    def stages(self):
        stages = [stage.clone() for stage in self._stages]
        for stage in stages:
            if stage.algorithm_key != "fieldscale":
                continue
            if self._fieldscale_temporal:
                stage.parameters.setdefault("temporal_smoothing_percent", 90)
            else:
                stage.parameters.pop("temporal_smoothing_percent", None)
        return stages

    def enabled_stages(self):
        return [stage for stage in self.stages() if stage.enabled]

    def configuration(self):
        return [stage.to_dict() for stage in self.stages()]

    def summary(self):
        return pipeline_summary_text(self._stages)

    def set_stages(self, stages, selected_row=0):
        self._stages = [
            stage.clone()
            if isinstance(stage, AlgorithmStage)
            else AlgorithmStage(
                stage["algorithm"],
                stage.get("enabled", True),
                stage.get("parameters", {}),
            )
            for stage in stages
        ]
        self.refresh_stage_list(selected_row)

    def parameter_summary(self, stage):
        params = stage.parameters
        if stage.algorithm_key == "histogram_weighted":
            gain = params["histogram_weighted_gain_limit"]
            return (
                f"grid={params['histogram_weighted_grid']}, clip={params['histogram_weighted_clip']:g}, "
                f"bright={params['histogram_weighted_bright_weight']:g}, Gain Limit={gain or 'Off'}"
            )
        if stage.algorithm_key == "auto_contrast":
            return (
                f"P{params['low_percentile']:.1f}/P{params['high_percentile']:.1f}; "
                f"Gain Limit={params.get('percentile_gain_limit', 0) or 'Off'}"
            )
        if stage.algorithm_key == "fieldscale":
            summary = (
                f"Grid={params.get('grid_count', 8)}x"
                f"{params.get('grid_count', 8)}; "
                f"LES d={params.get('les_distance', 2)}, "
                f"th={params.get('les_threshold_raw14', 400)}; "
                f"MP d={params.get('message_passing_distance', 1)}, "
                f"N={params.get('message_passing_iterations', 7)}"
            )
            max_gain = float(params.get("max_local_gain", 0.0))
            summary += (
                f"; MaxGain={max_gain:.0f}x"
                if max_gain >= 1.0
                else "; MaxGain=Off"
            )
            if self._fieldscale_temporal:
                summary += (
                    "; Temporal="
                    f"{params.get('temporal_smoothing_percent', 90)}%"
                )
            return summary
        if stage.algorithm_key == "clahe":
            return f"clip={params['clahe_clip']:.1f}, grid={params['clahe_grid']}"
        if stage.algorithm_key == "gamma":
            return f"gamma={params['gamma']:.1f}"
        if stage.algorithm_key == "median":
            size = params["median_kernel_size"]
            return f"kernel={size}x{size}"
        if stage.algorithm_key == "gaussian":
            size = params["gaussian_kernel_size"]
            return f"kernel={size}x{size}, sigma={params['gaussian_sigma']:.1f}"
        if stage.algorithm_key == "dde":
            return (
                f"sigma_s={params.get('dde_sigma_s', 21)}, "
                f"sigma_r={params.get('dde_sigma_r', 51)}, "
                f"gain={params.get('dde_gain_percent', 200)}%, "
                f"iterations={params.get('dde_num_iters', 3)}"
            )
        if stage.algorithm_key == "bilateral":
            return f"strength={params['bilateral_strength']}"
        if stage.algorithm_key == "unsharp":
            return f"amount={params['unsharp_amount']:.1f}"
        return "no parameters"

    def refresh_stage_list(self, selected_row=None):
        if selected_row is None:
            selected_row = self.stage_list.currentRow()
        self._updating = True
        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        for index, stage in enumerate(self._stages, start=1):
            text = (
                f"{index}. {algorithm_display_name(stage.algorithm_key)}"
                f"    [{self.parameter_summary(stage)}]"
            )
            item = QListWidgetItem(text)
            item.setToolTip(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if stage.enabled else Qt.Unchecked)
            self.stage_list.addItem(item)
        self.stage_list.blockSignals(False)
        self._updating = False
        if self._stages:
            self.stage_list.setCurrentRow(
                min(max(selected_row, 0), len(self._stages) - 1)
            )
        else:
            self.load_selected_step(-1)
        self.update_button_states()

    def add_stage(self, _checked=False):
        self._stages.append(AlgorithmStage(self.add_algorithm_combo.currentData()))
        self.refresh_stage_list(len(self._stages) - 1)
        self.pipeline_changed.emit()

    def remove_selected(self, _checked=False):
        row = self.stage_list.currentRow()
        if row < 0:
            return
        del self._stages[row]
        self.refresh_stage_list(min(row, len(self._stages) - 1))
        self.pipeline_changed.emit()

    def move_selected(self, offset):
        row = self.stage_list.currentRow()
        target = row + offset
        if row < 0 or target < 0 or target >= len(self._stages):
            return
        self._stages[row], self._stages[target] = self._stages[target], self._stages[row]
        self.refresh_stage_list(target)
        self.pipeline_changed.emit()

    def reset_selected_step(self, _checked=False):
        row = self.stage_list.currentRow()
        if row < 0:
            return
        self._stages[row].parameters = default_algorithm_parameters(
            self._stages[row].algorithm_key
        )
        self.refresh_stage_list(row)
        self.pipeline_changed.emit()

    def reset_all_parameters(self, _checked=False):
        if not self._stages:
            return
        for stage in self._stages:
            stage.parameters = default_algorithm_parameters(stage.algorithm_key)
        self.refresh_stage_list()
        self.pipeline_changed.emit()

    def reset_pipeline(self, _checked=False):
        self.set_stages([])
        self.pipeline_changed.emit()

    def stage_check_changed(self, item):
        if self._updating:
            return
        row = self.stage_list.row(item)
        if 0 <= row < len(self._stages):
            self._stages[row].enabled = item.checkState() == Qt.Checked
            self.pipeline_changed.emit()

    def selected_algorithm_changed(self, _combo_index):
        if self._updating:
            return
        row = self.stage_list.currentRow()
        if row < 0:
            return
        self._stages[row] = AlgorithmStage(
            self.stage_algorithm_combo.currentData(),
            enabled=self._stages[row].enabled,
        )
        self.refresh_stage_list(row)
        self.pipeline_changed.emit()

    def selected_parameter_changed(self, _value=None):
        if self._updating:
            return
        row = self.stage_list.currentRow()
        if row < 0:
            return
        self.update_percentile_limits()
        stage = self._stages[row]
        if stage.algorithm_key == "auto_contrast":
            stage.parameters["low_percentile"] = self.stage_low_percentile.value()
            stage.parameters["high_percentile"] = self.stage_high_percentile.value()
            stage.parameters["percentile_gain_limit"] = self.stage_percentile_gain_limit.value()
        elif stage.algorithm_key == "histogram_weighted":
            for key, widget in self.histogram_weighted_widgets.items():
                if isinstance(widget, QCheckBox):
                    stage.parameters[key] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    stage.parameters[key] = widget.currentData()
                else:
                    stage.parameters[key] = widget.value()
        elif stage.algorithm_key == "fieldscale":
            stage.parameters["grid_count"] = (
                self.stage_fieldscale_grid_count.currentData()
            )
            stage.parameters["les_distance"] = (
                self.stage_fieldscale_les_distance.value()
            )
            stage.parameters["les_threshold_raw14"] = (
                self.stage_fieldscale_les_threshold.value()
            )
            stage.parameters["message_passing_distance"] = (
                self.stage_fieldscale_message_distance.value()
            )
            stage.parameters["message_passing_iterations"] = (
                self.stage_fieldscale_message_iterations.value()
            )
            stage.parameters["max_local_gain"] = float(
                self.stage_fieldscale_max_local_gain.currentData()
            )
            if self._fieldscale_temporal:
                stage.parameters["temporal_smoothing_percent"] = (
                    self.stage_fieldscale_temporal_smoothing.value()
                )
            else:
                stage.parameters.pop("temporal_smoothing_percent", None)
        elif stage.algorithm_key == "clahe":
            stage.parameters["clahe_clip"] = self.stage_clahe_clip.value()
            stage.parameters["clahe_grid"] = self.stage_clahe_grid.value()
        elif stage.algorithm_key == "gamma":
            stage.parameters["gamma"] = self.stage_gamma.value()
        elif stage.algorithm_key == "median":
            stage.parameters["median_kernel_size"] = (
                self.stage_median_kernel_size.currentData()
            )
        elif stage.algorithm_key == "gaussian":
            stage.parameters["gaussian_kernel_size"] = (
                self.stage_gaussian_kernel_size.currentData()
            )
            stage.parameters["gaussian_sigma"] = self.stage_gaussian_sigma.value()
        elif stage.algorithm_key == "dde":
            stage.parameters["dde_sigma_s"] = self.stage_dde_sigma_s.value()
            stage.parameters["dde_sigma_r"] = self.stage_dde_sigma_r.value()
            stage.parameters["dde_gain_percent"] = self.stage_dde_gain_percent.value()
            stage.parameters["dde_num_iters"] = self.stage_dde_num_iters.value()
        elif stage.algorithm_key == "bilateral":
            stage.parameters["bilateral_strength"] = self.stage_bilateral_strength.value()
        elif stage.algorithm_key == "unsharp":
            stage.parameters["unsharp_amount"] = self.stage_unsharp_amount.value()
        self.refresh_stage_list(row)
        self.pipeline_changed.emit()

    def selected_fieldscale_grid_changed(self, _index=None):
        if self._updating:
            return
        self._updating = True
        self.update_fieldscale_distance_limits()
        self._updating = False
        self.selected_parameter_changed()

    def update_fieldscale_distance_limits(self):
        grid_count = self.stage_fieldscale_grid_count.currentData()
        maximum_distance = max(1, int(grid_count or 8) - 1)
        self.stage_fieldscale_les_distance.setMaximum(maximum_distance)
        self.stage_fieldscale_message_distance.setMaximum(maximum_distance)
        local_controls_enabled = int(grid_count or 8) > 1
        for widget in (
            self.stage_fieldscale_les_distance,
            self.stage_fieldscale_les_threshold,
            self.stage_fieldscale_message_distance,
            self.stage_fieldscale_message_iterations,
        ):
            widget.setEnabled(local_controls_enabled)

    def set_parameter_visible(self, widget, visible):
        row = self.parameter_rows.get(widget)
        if row is not None:
            self.parameter_group.set_body_widget_visible(row, visible)
        else:
            self.parameter_group.set_body_widget_visible(widget, visible)
        self.parameter_group.updateGeometry()
        self.updateGeometry()

    def update_parameter_size_constraints(self, _expanded=None):
        """Keep parameter controls usable; let the containing form scroll instead."""
        self.parameter_group.setMinimumHeight(0)
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        if self.parameter_group.is_expanded():
            margins = self.parameter_form.contentsMargins()
            content_width = max(
                1, self.parameter_group.width() - margins.left() - margins.right()
            )
            description_height = self.stage_description.heightForWidth(
                content_width
            )
            if description_height < 0:
                description_height = self.stage_description.sizeHint().height()
            visible_widgets = [self.stage_description]
            visible_widgets.extend(
                row for row in self.parameter_rows.values() if not row.isHidden()
            )
            if not self.reset_step_button.isHidden():
                visible_widgets.append(self.reset_step_button)
            content_height = (
                description_height
                + sum(
                    widget.sizeHint().height()
                    for widget in visible_widgets[1:]
                )
                + margins.top()
                + margins.bottom()
                + self.parameter_form.spacing() * max(0, len(visible_widgets) - 1)
            )
            title_height = self.parameter_group._title_rect().height() + 16
            self.parameter_group.setMinimumHeight(title_height + content_height)
        self.layout().invalidate()
        self.setFixedHeight(max(
            self.layout().minimumSize().height(),
            self.layout().totalHeightForWidth(self.width()),
        ))
        self.updateGeometry()
        parent = self.parentWidget()
        if parent is not None and parent.layout() is not None:
            for index in range(parent.layout().count()):
                child_layout = parent.layout().itemAt(index).layout()
                if child_layout is not None:
                    child_layout.invalidate()
            parent.layout().invalidate()
            if parent.layout().indexOf(self) >= 0:
                parent.setMinimumHeight(max(
                    parent.layout().minimumSize().height(),
                    parent.layout().totalHeightForWidth(parent.width()),
                ))
            parent.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._parameter_size_update_pending:
            return
        self._parameter_size_update_pending = True

        def refresh_parameter_height():
            self._parameter_size_update_pending = False
            self.update_parameter_size_constraints()

        QTimer.singleShot(0, refresh_parameter_height)

    def hasHeightForWidth(self):
        # Height is maintained by update_parameter_size_constraints. Let the
        # outer form use that minimum instead of a competing wrapped-text hint.
        return False

    def load_selected_step(self, row):
        from thermal_toolkit.algorithms import weighted_aci
        automatic_clip = weighted_aci.ACTIVE_WEIGHTED_ACI['enabled']
        self.stage_histogram_weighted_clip.setEnabled(not automatic_clip)
        self.stage_histogram_weighted_clip.setToolTip(
            "Automatic Clip enabled: configure thresholds in Setting > Weighted mapping."
            if automatic_clip else "Manual Clip x pixel count / Unique Levels; 0 disables clipping.")
        valid = 0 <= row < len(self._stages)
        self.parameter_group.set_body_widget_visible(self.reset_step_button, valid)
        self.set_parameter_visible(
            self.stage_algorithm_combo, valid and self._allow_step_algorithm_change
        )
        self._updating = True
        if valid:
            stage = self._stages[row]
            index = self.stage_algorithm_combo.findData(stage.algorithm_key)
            self.stage_algorithm_combo.setCurrentIndex(max(index, 0))
            params = stage.parameters
            for key, default in default_algorithm_parameters("histogram_weighted").items():
                widget = self.histogram_weighted_widgets[key]
                value = params.get(key, default)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentIndex(max(0, widget.findData(value)))
                else:
                    widget.setValue(value)
                self.set_parameter_visible(widget, stage.algorithm_key == "histogram_weighted")
            self.stage_histogram_weighted_redistribute.setEnabled(automatic_clip or self.stage_histogram_weighted_clip.value() > 0)
            self.stage_histogram_weighted_bright_percentile.setEnabled(self.stage_histogram_weighted_bright_weight.currentData() != 1.0)
            self.stage_low_percentile.setValue(params.get("low_percentile", 1.0))
            self.stage_high_percentile.setValue(params.get("high_percentile", 99.0))
            self.stage_percentile_gain_limit.setValue(params.get("percentile_gain_limit", 0.0))
            self.stage_clahe_clip.setValue(params.get("clahe_clip", 2.0))
            self.stage_clahe_grid.setValue(params.get("clahe_grid", 8))
            self.stage_gamma.setValue(params.get("gamma", 1.5))
            median_kernel_index = self.stage_median_kernel_size.findData(
                params.get("median_kernel_size", 3)
            )
            self.stage_median_kernel_size.setCurrentIndex(
                median_kernel_index if median_kernel_index >= 0 else 0
            )
            gaussian_kernel_index = self.stage_gaussian_kernel_size.findData(
                params.get("gaussian_kernel_size", 5)
            )
            self.stage_gaussian_kernel_size.setCurrentIndex(
                gaussian_kernel_index if gaussian_kernel_index >= 0 else 1
            )
            self.stage_gaussian_sigma.setValue(params.get("gaussian_sigma", 1.0))
            self.stage_bilateral_strength.setValue(params.get("bilateral_strength", 2))
            self.stage_unsharp_amount.setValue(params.get("unsharp_amount", 1.0))
            self.stage_dde_sigma_s.setValue(params.get("dde_sigma_s", 21))
            self.stage_dde_sigma_r.setValue(params.get("dde_sigma_r", 51))
            self.stage_dde_gain_percent.setValue(params.get("dde_gain_percent", 200))
            self.stage_dde_num_iters.setValue(params.get("dde_num_iters", 3))
            grid_index = self.stage_fieldscale_grid_count.findData(
                params.get("grid_count", 8)
            )
            self.stage_fieldscale_grid_count.setCurrentIndex(
                grid_index
                if grid_index >= 0
                else self.stage_fieldscale_grid_count.findData(8)
            )
            self.update_fieldscale_distance_limits()
            self.stage_fieldscale_les_distance.setValue(
                params.get("les_distance", 2)
            )
            self.stage_fieldscale_les_threshold.setValue(
                params.get("les_threshold_raw14", 400)
            )
            self.stage_fieldscale_message_distance.setValue(
                params.get("message_passing_distance", 1)
            )
            self.stage_fieldscale_message_iterations.setValue(
                params.get("message_passing_iterations", 7)
            )
            max_gain_index = self.stage_fieldscale_max_local_gain.findData(
                float(params.get("max_local_gain", 0.0))
            )
            self.stage_fieldscale_max_local_gain.setCurrentIndex(
                max_gain_index if max_gain_index >= 0 else 0
            )
            self.stage_fieldscale_temporal_smoothing.setValue(
                params.get("temporal_smoothing_percent", 90)
            )
            self.stage_description.setText(
                IMAGE_ALGORITHM_DESCRIPTIONS.get(stage.algorithm_key, "")
            )
            is_fieldscale = stage.algorithm_key == "fieldscale"
            for widget in (
                self.stage_fieldscale_grid_count,
                self.stage_fieldscale_les_distance,
                self.stage_fieldscale_les_threshold,
                self.stage_fieldscale_message_distance,
                self.stage_fieldscale_message_iterations,
                self.stage_fieldscale_max_local_gain,
            ):
                self.set_parameter_visible(widget, is_fieldscale)
            self.set_parameter_visible(
                self.stage_fieldscale_temporal_smoothing,
                is_fieldscale and self._fieldscale_temporal,
            )
            self.set_parameter_visible(
                self.stage_low_percentile, stage.algorithm_key == "auto_contrast"
            )
            self.set_parameter_visible(
                self.stage_high_percentile, stage.algorithm_key == "auto_contrast"
            )
            self.set_parameter_visible(self.stage_percentile_gain_limit, stage.algorithm_key == "auto_contrast")
            self.set_parameter_visible(
                self.stage_clahe_clip, stage.algorithm_key == "clahe"
            )
            self.set_parameter_visible(
                self.stage_clahe_grid, stage.algorithm_key == "clahe"
            )
            self.set_parameter_visible(self.stage_gamma, stage.algorithm_key == "gamma")
            self.set_parameter_visible(
                self.stage_median_kernel_size, stage.algorithm_key == "median"
            )
            self.set_parameter_visible(
                self.stage_gaussian_kernel_size, stage.algorithm_key == "gaussian"
            )
            self.set_parameter_visible(
                self.stage_gaussian_sigma, stage.algorithm_key == "gaussian"
            )
            for widget in (
                self.stage_dde_sigma_s,
                self.stage_dde_sigma_r,
                self.stage_dde_gain_percent,
                self.stage_dde_num_iters,
            ):
                self.set_parameter_visible(
                    widget, stage.algorithm_key == "dde"
                )
            self.set_parameter_visible(
                self.stage_bilateral_strength, stage.algorithm_key == "bilateral"
            )
            self.set_parameter_visible(
                self.stage_unsharp_amount, stage.algorithm_key == "unsharp"
            )
            self.update_percentile_limits()
        else:
            for parameter_row in self.parameter_rows.values():
                self.parameter_group.set_body_widget_visible(parameter_row, False)
            self.stage_description.setText(
                "Waiting for an algorithm. Select an algorithm above and click Add Step."
            )
        self.parameter_group.updateGeometry()
        self.update_parameter_size_constraints()
        self.updateGeometry()
        self._updating = False
        self.update_button_states()

    def update_percentile_limits(self):
        self.stage_high_percentile.setMinimum(
            min(100.0, self.stage_low_percentile.value() + 0.5)
        )
        self.stage_low_percentile.setMaximum(
            max(0.0, self.stage_high_percentile.value() - 0.5)
        )

    def update_button_states(self):
        row = self.stage_list.currentRow()
        valid = 0 <= row < len(self._stages)
        self.move_up_button.setEnabled(valid and row > 0)
        self.move_down_button.setEnabled(valid and row < len(self._stages) - 1)
        self.remove_button.setEnabled(valid)
        self.reset_step_button.setEnabled(valid)
        self.reset_all_parameters_button.setEnabled(bool(self._stages))
