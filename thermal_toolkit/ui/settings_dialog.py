"""ui / settings_dialog."""

from thermal_toolkit.core.constants import BIT_DEPTH_OPTIONS, RESOLUTION_OPTIONS
from thermal_toolkit.core.qt import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSignal,
)
from thermal_toolkit.core.settings import default_toolkit_settings, validated_toolkit_settings
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox, FocusWheelSpinBox, FocusWheelDoubleSpinBox, PathRow


class ToolkitSettingsDialog(QDialog):
    settings_applied = pyqtSignal(object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setObjectName("toolkitSettingsDialog")
        self.setWindowTitle("Toolkit Setting")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(720, 760)
        self.setMinimumSize(640, 680)
        self.setSizeGripEnabled(True)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        general_page = QWidget()
        general_form = QFormLayout(general_page)
        general_form.setContentsMargins(18, 18, 18, 18)
        general_form.setHorizontalSpacing(16)
        general_form.setVerticalSpacing(14)
        self.resolution_combo = FocusWheelComboBox()
        for width, height in RESOLUTION_OPTIONS:
            self.resolution_combo.addItem(
                f"{width} × {height}", (width, height)
            )
        self.bit_depth_combo = FocusWheelComboBox()
        for bits in BIT_DEPTH_OPTIONS:
            self.bit_depth_combo.addItem(
                f"{bits}-bit (Max {(1 << bits) - 1})", bits
            )
        self.output_root_row = PathRow(
            "folder", dialog_title="Select Output Root Folder"
        )
        general_form.addRow("Resolution", self.resolution_combo)
        general_form.addRow("Bit Depth", self.bit_depth_combo)
        general_form.addRow("Output Root", self.output_root_row)
        self.camera_a_mirror_check = QCheckBox("Mirror Camera A horizontally")
        self.camera_b_mirror_check = QCheckBox("Mirror Camera B horizontally")
        mirror_options = QWidget()
        mirror_layout = QVBoxLayout(mirror_options)
        mirror_layout.setContentsMargins(0, 0, 0, 0)
        mirror_layout.setSpacing(6)
        mirror_layout.addWidget(self.camera_a_mirror_check)
        mirror_layout.addWidget(self.camera_b_mirror_check)
        general_form.addRow("Camera Mirror", mirror_options)
        self.tabs.addTab(general_page, "General")

        weighted_page = QWidget()
        weighted_form = QFormLayout(weighted_page)
        weighted_form.setContentsMargins(18, 18, 18, 18)
        weighted_form.setVerticalSpacing(14)
        self.weighted_auto_clip_check = QCheckBox("Enable automatic Clip")
        weighted_form.addRow(self.weighted_auto_clip_check)
        self.weighted_parameter_mode_combo = FocusWheelComboBox()
        self.weighted_parameter_mode_combo.addItem("Two parameters: Range + Standard deviation", "dual")
        self.weighted_parameter_mode_combo.addItem("One parameter: (P99 - P1) / mean", "single")
        weighted_form.addRow("Parameters", self.weighted_parameter_mode_combo)
        self.weighted_ratio_mode_combo = FocusWheelComboBox()
        for label, mode in (("MIN", "min"), ("Average", "mean"), ("MAX", "max")):
            self.weighted_ratio_mode_combo.addItem(label, mode)
        self.weighted_ratio_mode_combo.setToolTip("Combine the normalized Range and Standard deviation ratios using MIN, Average or MAX.")
        self.weighted_ratio_mode_combo.currentIndexChanged.connect(self.clear_apply_status)
        self.weighted_ratio_mode_label = QLabel("Ratio Combination")
        weighted_form.addRow(self.weighted_ratio_mode_label, self.weighted_ratio_mode_combo)
        self.weighted_threshold_spins = {}
        self.weighted_threshold_labels = {}
        for threshold, title in (("thl", "THL (Low Thresholds)"),
                                 ("thh", "THH (High Thresholds)")):
            group = QGroupBox(title)
            group.setObjectName("settingSection")
            group_form = QFormLayout(group)
            group_form.setVerticalSpacing(10)
            for name, label in (("range", "Range (P99 - P1)"),
                                ("sigma", "Standard deviation"),
                                ("relative_range", "(P99 - P1) / mean")):
                key = name + "_" + threshold
                spin = FocusWheelDoubleSpinBox()
                spin.setRange(0.0, 65535.0)
                spin.setDecimals(4 if name == "relative_range" else 1)
                spin.setSingleStep(0.01 if name == "relative_range" else 1.0)
                spin.setToolTip("Measured in the input values of each Weighted mapping tile. THL must be smaller than THH.")
                self.weighted_threshold_spins[key] = spin
                if name == "relative_range":
                    spin.setToolTip("Dimensionless (P99 - P1) / mean per tile; zero mean uses 0. THL must be smaller than THH.")
                field_label = QLabel(label)
                self.weighted_threshold_labels[key] = field_label
                group_form.addRow(field_label, spin)
                spin.valueChanged.connect(self.clear_apply_status)
            weighted_form.addRow(group)
        self.weighted_parameter_mode_combo.currentIndexChanged.connect(self.update_weighted_parameter_mode)
        note = QLabel("Two parameters: normalize Range and standard deviation, then MIN/Average/MAX.\n"
                      "One parameter: normalize (P99 - P1) / mean with its own THL/THH.\n"
                      "Mean and standard deviation use all tile pixels; zero mean uses 0.\n"
                      "Clip = 20 + ratio * (2 * pixels / UniqueLevels - 20).\n"
                      "Grid > 1 computes statistics per tile. Dual thresholds use input DN;\n"
                      "8-bit Enhancement uses 0..255 input values.\n"
                      "Automatic Clip overrides the manual Clip value. Apply saves settings.")
        note.setWordWrap(True)
        weighted_form.addRow(note)
        self.weighted_auto_clip_check.toggled.connect(self.clear_apply_status)
        self.tabs.addTab(weighted_page, "Weighted mapping")

        calibration_page = QWidget()
        calibration_form = QFormLayout(calibration_page)
        calibration_form.setContentsMargins(18, 18, 18, 18)
        calibration_form.setVerticalSpacing(14)
        self.bad_pixel_spins = {}
        for key, title in (('dc_sigma', 'DC threshold (local sigma)'),
                           ('response_percent', 'Responsivity deviation (%)'),
                           ('netd_sigma', 'NETD threshold (global sigma)'),
                           ('window', 'DC neighbourhood (odd width)')):
            spin = FocusWheelSpinBox() if key == 'window' else FocusWheelDoubleSpinBox()
            if key == 'window':
                spin.setRange(3,31)
                spin.setSingleStep(2)
            else:
                spin.setRange(.01, 1000 if key == 'response_percent' else 100)
            self.bad_pixel_spins[key] = spin
            calibration_form.addRow(title, spin)
            spin.valueChanged.connect(self.clear_apply_status)
        self.tabs.addTab(calibration_page, 'Bad pixel calibration')

        appearance_page = QWidget()
        appearance_form = QFormLayout(appearance_page)
        appearance_form.setContentsMargins(18, 18, 18, 18)
        appearance_form.setHorizontalSpacing(16)
        appearance_form.setVerticalSpacing(14)
        self.log_font_spin = FocusWheelSpinBox()
        self.log_font_spin.setRange(10, 28)
        self.log_font_spin.setSuffix(" px")
        appearance_form.addRow("Log Font Size", self.log_font_spin)
        self.tabs.addTab(appearance_page, "Appearance")

        self.raw_histogram_cdf_check = QCheckBox("Show CDF line")
        self.raw_histogram_cdf_check.setToolTip("顯示或隱藏 Image 與 UVC 直方圖的累積分布曲線；按 Apply 後生效。")
        appearance_form.addRow("Image / UVC Histogram", self.raw_histogram_cdf_check)
        self.raw_histogram_cdf_check.toggled.connect(self.clear_apply_status)

        self.apply_status = QLabel()
        self.apply_status.setAlignment(Qt.AlignCenter)
        self.apply_status.setMinimumHeight(28)
        root.addWidget(self.apply_status)

        button_row = QHBoxLayout()
        self.defaults_button = QPushButton("Reset to Default")
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primaryButton")
        close_button = QPushButton("Close")
        button_row.addWidget(self.defaults_button)
        button_row.addStretch()
        button_row.addWidget(self.apply_button)
        button_row.addWidget(close_button)
        root.addLayout(button_row)
        self.defaults_button.clicked.connect(self.reset_defaults)
        self.apply_button.clicked.connect(self.apply_settings)
        close_button.clicked.connect(self.close)
        self.resolution_combo.currentIndexChanged.connect(
            self.clear_apply_status
        )
        self.bit_depth_combo.currentIndexChanged.connect(
            self.clear_apply_status
        )
        self.output_root_row.edit.textChanged.connect(
            self.clear_apply_status
        )
        self.camera_a_mirror_check.toggled.connect(self.clear_apply_status)
        self.camera_b_mirror_check.toggled.connect(self.clear_apply_status)
        self.log_font_spin.valueChanged.connect(self.clear_apply_status)
        self.set_settings(settings)

    def set_settings(self, settings):
        settings = validated_toolkit_settings(settings)
        raw_input = settings["raw_input"]
        selected_resolution = tuple(raw_input["resolution"])
        resolution_index = next(
            (
                index
                for index in range(self.resolution_combo.count())
                if tuple(self.resolution_combo.itemData(index))
                == selected_resolution
            ),
            -1,
        )
        self.resolution_combo.setCurrentIndex(max(resolution_index, 0))
        bit_depth_index = self.bit_depth_combo.findData(
            raw_input["bit_depth"]
        )
        self.bit_depth_combo.setCurrentIndex(max(bit_depth_index, 0))
        self.output_root_row.setText(raw_input["output_root"])
        camera_mirror = settings["camera_mirror"]
        self.camera_a_mirror_check.setChecked(camera_mirror["camera_a"])
        self.camera_b_mirror_check.setChecked(camera_mirror["camera_b"])
        self.log_font_spin.setValue(settings["log_font_size"])
        timeline = settings["timeline"]
        self.raw_histogram_cdf_check.setChecked(timeline.get("raw_histogram_show_cdf", False))
        weighted = settings["weighted_mapping"]
        self.weighted_auto_clip_check.setChecked(weighted["enabled"])
        self.weighted_parameter_mode_combo.setCurrentIndex(self.weighted_parameter_mode_combo.findData(weighted['parameter_mode']))
        self.weighted_ratio_mode_combo.setCurrentIndex(self.weighted_ratio_mode_combo.findData(weighted['ratio_mode']))
        for key, spin in self.weighted_threshold_spins.items():
            spin.setValue(weighted[key])
        self.update_weighted_parameter_mode()
        for key, spin in self.bad_pixel_spins.items():
            spin.setValue(settings["bad_pixel_calibration"][key])
        self._settings = settings

    def current_settings(self):
        return dict(self._settings, **{
            "bad_pixel_calibration": {key: spin.value() for key, spin in self.bad_pixel_spins.items()},
            "weighted_mapping": dict(
                enabled=self.weighted_auto_clip_check.isChecked(),
                ratio_mode=self.weighted_ratio_mode_combo.currentData(),
                parameter_mode=self.weighted_parameter_mode_combo.currentData(),
                **{key: spin.value() for key, spin in self.weighted_threshold_spins.items()}),
            "log_font_size": self.log_font_spin.value(),
            "raw_input": {
                "resolution": list(self.resolution_combo.currentData()),
                "bit_depth": self.bit_depth_combo.currentData(),
                "output_root": self.output_root_row.text(),
            },
            "timeline": dict(self._settings["timeline"], **{
                "raw_histogram_show_cdf": self.raw_histogram_cdf_check.isChecked(),
            }),
            "camera_mirror": {
                "camera_a": self.camera_a_mirror_check.isChecked(),
                "camera_b": self.camera_b_mirror_check.isChecked(),
            },
        })

    def update_weighted_parameter_mode(self, _index=None):
        single = self.weighted_parameter_mode_combo.currentData() == "single"
        self.weighted_ratio_mode_combo.setVisible(not single)
        self.weighted_ratio_mode_label.setVisible(not single)
        for key, spin in self.weighted_threshold_spins.items():
            visible = key.startswith("relative_range_") == single
            spin.setVisible(visible)
            self.weighted_threshold_labels[key].setVisible(visible)
        self.clear_apply_status()

    def reset_defaults(self):
        self.set_settings(default_toolkit_settings())

    def clear_apply_status(self, _value=None):
        self.apply_status.clear()

    def show_apply_status(self, message, success=True):
        color = "#8ed0a2" if success else "#f0c078"
        self.apply_status.setText(message)
        self.apply_status.setStyleSheet(
            f"color:{color}; background:#202a25; border:1px solid {color}; "
            "border-radius:5px; padding:5px; font-size:14px; font-weight:700;"
        )

    def apply_settings(self):
        if not self.apply_button.isEnabled():
            return
        if self.bad_pixel_spins["window"].value() % 2 == 0:
            self.show_apply_status("DC neighbourhood width must be odd.", success=False)
            return
        names = ("relative_range",) if self.weighted_parameter_mode_combo.currentData() == "single" else ("range", "sigma")
        for name in names:
            if self.weighted_threshold_spins[name + "_thl"].value() >= self.weighted_threshold_spins[name + "_thh"].value():
                self.show_apply_status(f"{name}: THL must be smaller than THH.", success=False)
                return
        self.settings_applied.emit(self.current_settings())
