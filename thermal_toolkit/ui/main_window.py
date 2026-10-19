"""ui / main_window."""

from pathlib import Path
import os
import sys
from thermal_toolkit.analysis.statistics import image_stats
from thermal_toolkit.core.constants import (
    BIT_DEPTH_OPTIONS,
    CAMERA_REFRESH_INTERVAL_MS,
    DEFAULT_BIT_DEPTH,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    RESOLUTION_OPTIONS,
    UVC_CLOSE_TIMEOUT_MS,
)
from thermal_toolkit.core.paths import unique_result_dir
from thermal_toolkit.core.qt import (
    QAction,
    QActionGroup,
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
)
from thermal_toolkit.core.settings import load_toolkit_settings
from thermal_toolkit.core.version import APP_TITLE
from thermal_toolkit.features.average_raw.page import AverageRawPageMixin
from thermal_toolkit.features.bpr.page import BprPageMixin
from thermal_toolkit.features.bpr.calibration_page import BadPixelCalibrationPageMixin
from thermal_toolkit.features.dual_image.page import DualImagePageMixin
from thermal_toolkit.features.dual_uvc.page import DualUvcPageMixin
from thermal_toolkit.features.dual_uvc.recording import DualUvcRecordingMixin
from thermal_toolkit.features.high_low.page import HighLowPageMixin
from thermal_toolkit.features.image_algorithm.page import ImageAlgorithmPageMixin
from thermal_toolkit.features.image_algorithm.statistics import ImageAlgorithmStatisticsMixin
from thermal_toolkit.features.image_statistics.page import ImageStatisticsPageMixin
from thermal_toolkit.features.nuc.analysis_page import NucAnalysisPageMixin
from thermal_toolkit.features.nuc.calculation_page import NucCalculationPageMixin
from thermal_toolkit.features.single_uvc.page import SingleUvcPageMixin
from thermal_toolkit.features.single_uvc.recording import SingleUvcRecordingMixin
from thermal_toolkit.features.temporal_drift.page import TemporalDriftPageMixin
from thermal_toolkit.features.uvc_algorithm.page import UvcAlgorithmPageMixin
from thermal_toolkit.features.uvc_algorithm.recording import UvcAlgorithmRecordingMixin
from thermal_toolkit.ui.camera_controls import CameraControlsMixin
from thermal_toolkit.ui.image_analysis import ImageAnalysisMixin
from thermal_toolkit.ui.preview_windows import PreviewWindowsMixin
from thermal_toolkit.ui.quality_controls import QualityControlsMixin
from thermal_toolkit.ui.settings import SettingsMixin
from thermal_toolkit.ui.theme import ThemeMixin
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox, PathRow
from thermal_toolkit.ui.widgets.preview import PreviewLabel, SwipeCompareWidget
from thermal_toolkit.ui.widgets.title_bar import ToolkitTitleBar
from thermal_toolkit.uvc.buffer import LatestFrameBuffer


class ThermalRawToolkit(
    ThemeMixin,
    SettingsMixin,
    QualityControlsMixin,
    NucCalculationPageMixin,
    BprPageMixin,
    BadPixelCalibrationPageMixin,
    ImageAlgorithmPageMixin,
    ImageAnalysisMixin,
    PreviewWindowsMixin,
    DualImagePageMixin,
    SingleUvcPageMixin,
    ImageStatisticsPageMixin,
    NucAnalysisPageMixin,
    ImageAlgorithmStatisticsMixin,
    AverageRawPageMixin,
    HighLowPageMixin,
    TemporalDriftPageMixin,
    CameraControlsMixin,
    DualUvcPageMixin,
    UvcAlgorithmPageMixin,
    SingleUvcRecordingMixin,
    DualUvcRecordingMixin,
    UvcAlgorithmRecordingMixin,
    QMainWindow,
):
    FUNCTIONS = [
        "Single Image Statistics",
        "Average RAW Folder",
        "NUC Calculate",
        "NUC_Analyze",
        "High / Low Analyzer",
        "Temporal Drift Analyzer",
        "Bad pixel calibration",
        "BPR Bad Pixel Marking",
        "Dual Image Compare",
        "Single UVC Statistic",
        "Dual UVC Compare",
        "Image Algorithm Compare",
        "UVC Algorithm Compare",
    ]
    FUNCTION_GROUPS = [
        (
            "Calibration",
            (
                "Average RAW Folder",
                "NUC Calculate",
                "Bad pixel calibration",
            ),
        ),
        (
            "Analyzer",
            (
                "NUC_Analyze",
                "BPR Bad Pixel Marking",
                "High / Low Analyzer",
                "Temporal Drift Analyzer",
            ),
        ),
        (
            "Image",
            (
                "Single Image Statistics",
                "Dual Image Compare",
                "Image Algorithm Compare",
            ),
        ),
        (
            "UVC",
            (
                "Single UVC Statistic",
                "Dual UVC Compare",
                "UVC Algorithm Compare",
            ),
        ),
    ]
    FUNCTION_DESCRIPTIONS = [
        "選擇 RAW 檔後立即顯示影像、RAW DN 統計、Histogram／CDF 與單張品質評分；Run／Save Analysis 會保存完整分析結果。",
        "將資料夾內的 RAW 影像逐像素平均，並依 LOW／HIGH 後綴儲存平均結果。",
        "使用 LOW 與 HIGH RAW 計算 NUC 參數，產生 Novatek pointN_K.bin 與 pointN_B.bin；右側以固定 Bit Depth 比例驗證校正後 HIGH。",
        "Load K, B and RAW for live original/corrected statistics. Adjust N and press Save to export analysis.",
        "逐像素比較 LOW 與 HIGH RAW，影像區顯示 HIGH - LOW Difference，並分析 DN 差異及異常像素。",
        "分析資料夾內多張 RAW 的時間變化，計算標準差與 Peak-to-Peak。",
        "讀取單張 RAW，選擇輸入映射，再依自訂數量與順序執行影像演算法 Pipeline。",
        "開啟 UVC Camera：預設 Auto Detect 先嘗試 Novatek MJPG，失敗自動改用相機原生格式，再依自訂順序即時執行 Pipeline。",
    ]
    PREVIEW_LAYOUTS = [
        (False, "", "RAW Preview (Fixed Scale)"),
        (False, "", "Average RAW (Fixed Scale)"),
        (True, "HIGH Input (Fixed Scale)", "NUC Corrected HIGH (Fixed Scale)"),
        (True, "Original RAW (Fixed Scale)", "NUC Corrected RAW (Fixed Scale)"),
        (False, "", "HIGH - LOW Difference (Fixed Scale)"),
        (False, "", "Temporal Standard Deviation (Fixed Scale)"),
        (True, "Temperature A RAW (Fixed Scale)", "Bad Pixel Candidates (A for DC only; otherwise B)"),
        (True, "RAW Fixed-Scale / White Canvas", "BPR Bad Pixels Marked"),
        (True, "RAW Image A", "RAW Image B"),
        (False, "", "Single UVC Statistic"),
        (True, "UVC Camera A", "UVC Camera B"),
        (True, "Original RAW (Fixed Scale)", "Final Pipeline Output"),
        (True, "Original UVC", "Final Pipeline Output"),
    ]
    FUNCTION_DESCRIPTIONS.insert(
        6,
        "Load a Novatek BPR dpc_table.bin or x,y coordinate file. Optionally overlay bad-pixel circles on a RAW image; otherwise use a white canvas.",
    )
    FUNCTION_DESCRIPTIONS.insert(
        7,
        "Compare two same-scene images with Image Quality: native-bit-depth Base/Detail analysis, gray-level/connected flat-region separation, robust typical/worst noise, uniformity, clipping, and artifact scores.",
    )
    FUNCTION_DESCRIPTIONS.insert(
        8,
        "Open one UVC camera with Auto Detect for a clean live view. Double-click the image to open a resizable and maximizable image-only window.",
    )
    FUNCTION_DESCRIPTIONS.insert(
        9,
        "Open two independent UVC cameras with per-camera Auto Detect for side-by-side monitoring, paired capture, and synchronized comparison recording.",
    )
    FUNCTION_DESCRIPTIONS.insert(
        6,
        "Load two RAW captures at different temperatures. Run detects exactly unchanged DN pixels and generates Novatek dpc_table.bin.",
    )
    FUNCTION_DESCRIPTIONS[FUNCTIONS.index("Image Algorithm Compare")] = (
        "Select a RAW file to immediately compare the original image with the "
        "selected pipeline output. Algorithm and mapping changes refresh the "
        "preview automatically; Save Analysis Results writes the current analysis."
    )
    FUNCTION_DESCRIPTIONS[FUNCTIONS.index("Temporal Drift Analyzer")] = (
        "選擇 RAW 資料夾後自動分析時間變化；左側顯示統計，按 Save 才儲存資料與圖表。"
    )

    def _create_pages(self):
        self.build_image_statistics_page()
        self.build_average_raw_page()
        self.build_nuc_calculation_page()
        self.build_nuc_analysis_page()
        self.build_high_low_page()
        self.build_temporal_drift_page()
        self.build_bad_pixel_calibration_page()
        self.build_bpr_page()
        self.build_dual_image_page()
        self.build_single_uvc_page()
        self.build_dual_uvc_page()
        self.build_image_algorithm_page()
        self.build_uvc_algorithm_page()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1450, 900)
        self.active_function_index = 0
        self.app_settings = load_toolkit_settings()
        self.settings_dialog = None
        self.quality_help_dialogs = {}
        self.quality_timeline_dialogs = {}
        self.current_original = None
        self.current_result = None
        self.last_output_dir = None
        self.uvc_video_window = None
        self.viewer_uvc_histogram_window = None
        self.raw_viewer_histogram_window = None
        self.raw_viewer_stats = None
        self.raw_viewer_histogram = None
        self.raw_viewer_histogram_rgb = None
        self.raw_viewer_quality = None
        self.raw_quality_result_a = None
        self.raw_quality_result_b = None
        self.camera_devices = []
        self._camera_device_signature = None
        self.viewer_uvc_worker = None
        self.viewer_uvc_frame = None
        self.viewer_uvc_actual_fps = 0.0
        self.viewer_uvc_output_fps = 0.0
        self.viewer_uvc_frame_count = 0
        self.viewer_uvc_fps_started = 0.0
        self.viewer_uvc_last_analysis_time = 0.0
        self.viewer_uvc_stats = None
        self.viewer_uvc_histogram = None
        self.viewer_uvc_histogram_rgb = None
        self.viewer_uvc_restart_when_stopped = False
        self.viewer_uvc_recording_worker = None
        self.viewer_uvc_recording_output_dir = None
        self.viewer_uvc_recording_manifest = None
        self.viewer_uvc_recording_start_histogram = None
        self.viewer_uvc_recording_stop_reason = None
        self.viewer_uvc_recording_started_perf = 0.0
        self.viewer_uvc_buffer = LatestFrameBuffer()
        self.viewer_uvc_buffer_sequence = 0
        self.viewer_uvc_display_timer = QTimer(self)
        self.viewer_uvc_display_timer.setInterval(10)
        self.viewer_uvc_display_timer.timeout.connect(
            self.consume_uvc_viewer_latest_frame
        )
        self.raw_algorithm_source = None
        self.raw_pipeline_input = None
        self.raw_pipeline_input_max = 255
        self.raw_pipeline_final = None
        self.raw_pipeline_records = []
        self.raw_algorithm_quality_original = None
        self.raw_algorithm_quality_final = None
        self.dual_uvc_worker_a = None
        self.dual_uvc_worker_b = None
        self.dual_uvc_frame_a = None
        self.dual_uvc_frame_b = None
        self.dual_uvc_frame_time_a = 0.0
        self.dual_uvc_frame_time_b = 0.0
        self.dual_uvc_actual_fps_a = 0.0
        self.dual_uvc_actual_fps_b = 0.0
        self.dual_uvc_output_fps_a = 0.0
        self.dual_uvc_output_fps_b = 0.0
        self.dual_uvc_frame_count_a = 0
        self.dual_uvc_frame_count_b = 0
        self.dual_uvc_fps_started_a = 0.0
        self.dual_uvc_fps_started_b = 0.0
        self.dual_uvc_recording_worker = None
        self.dual_uvc_recording_output_dir = None
        self.dual_uvc_recording_manifest = None
        self.dual_uvc_recording_stop_reason = None
        self.dual_uvc_recording_started_perf = 0.0
        self.dual_uvc_restart_when_stopped = False
        self.dual_uvc_close_pending = False
        self.dual_uvc_buffer_a = LatestFrameBuffer()
        self.dual_uvc_buffer_b = LatestFrameBuffer()
        self.dual_uvc_buffer_sequence_a = 0
        self.dual_uvc_buffer_sequence_b = 0
        self.dual_uvc_display_timer = QTimer(self)
        self.dual_uvc_display_timer.setInterval(10)
        self.dual_uvc_display_timer.timeout.connect(
            self.consume_dual_uvc_latest_frames
        )
        self.uvc_worker = None
        self.uvc_processing_worker = None
        self.uvc_recording_worker = None
        self.uvc_recording_output_dir = None
        self.uvc_recording_manifest = None
        self.uvc_recording_stop_reason = None
        self.uvc_recording_started_perf = 0.0
        self.uvc_actual_fps = 0.0
        self.uvc_original_source = None
        self.uvc_original_frame = None
        self.uvc_processed_frame = None
        self.uvc_pipeline_input = None
        self.uvc_pipeline_input_max = 255
        self.uvc_pipeline_records = []
        self.uvc_pipeline_total_ms = 0.0
        self.uvc_processing_revision = 0
        self.uvc_last_processing_error = None
        self.uvc_frame_count = 0
        self.uvc_fps_display = 0.0
        self.uvc_last_time = 0.0
        self.uvc_last_frame_time = 0.0
        self.uvc_waiting_for_frame = False
        self.uvc_reconnect_requested = False
        self.uvc_restart_when_stopped = False
        self.uvc_close_pending = False
        self.uvc_watchdog_timer = QTimer(self)
        self.uvc_watchdog_timer.setInterval(200)
        self.uvc_watchdog_timer.timeout.connect(self.check_uvc_stream)
        self._build_ui()
        self._apply_ui_style()
        self.title_bar = ToolkitTitleBar(self)
        self.setMenuWidget(self.title_bar)
        self._apply_category_badge_font()
        self.apply_toolkit_settings(
            self.app_settings, persist=False, announce=False
        )
        self.refresh_camera_devices(log_changes=False)
        self.camera_refresh_timer = QTimer(self)
        self.camera_refresh_timer.setInterval(CAMERA_REFRESH_INTERVAL_MS)
        self.camera_refresh_timer.timeout.connect(self.refresh_camera_devices)
        self.camera_refresh_timer.start()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        top = QHBoxLayout()
        self.category_badge = QPushButton(self.FUNCTION_GROUPS[0][0])
        self.category_badge.setObjectName("categoryBadge")
        self.category_badge.setCursor(Qt.PointingHandCursor)
        self.category_badge.setToolTip(
            "Select a Category and Function"
        )
        self.navigation_menu = QMenu(self.category_badge)
        self.navigation_action_group = QActionGroup(self)
        self.navigation_action_group.setExclusive(True)
        self.function_actions = {}
        for category_name, function_names in self.FUNCTION_GROUPS:
            category_menu = self.navigation_menu.addMenu(category_name)
            for function_name in function_names:
                function_index = self.FUNCTIONS.index(function_name)
                action = QAction(function_name, self)
                action.setCheckable(True)
                action.triggered.connect(
                    lambda _checked=False, index=function_index: (
                        self.select_function_from_menu(index)
                    )
                )
                self.navigation_action_group.addAction(action)
                self.function_actions[function_index] = action
                category_menu.addAction(action)
        self.category_badge.setMenu(self.navigation_menu)
        top.addWidget(self.category_badge)
        self.function_title = QLabel(self.FUNCTION_GROUPS[0][1][0])
        self.function_title.setStyleSheet("font-size:26px; font-weight:700; color:#86bce2;")
        top.addWidget(self.function_title)
        top.addStretch()
        root.addLayout(top)

        settings = QGroupBox()
        settings.setObjectName("settingsCard")
        sg = QVBoxLayout(settings)
        sg.setContentsMargins(14, 14, 14, 14)
        sg.setSpacing(10)
        self.function_description = QLabel()
        self.function_description.setWordWrap(True)
        self.function_description.setStyleSheet("font-size:18px; font-weight:600; color:#e6edf2; background:#2b323a; border-left:4px solid #79afd3; border-radius:6px; padding:10px 12px;")
        sg.addWidget(self.function_description)
        self.global_input_controls = QWidget(self)
        self.global_input_controls.hide()
        self.resolution_combo = FocusWheelComboBox(
            self.global_input_controls
        )
        for width, height in RESOLUTION_OPTIONS:
            self.resolution_combo.addItem(f"{width} × {height}", (width, height))
        default_resolution = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        default_index = RESOLUTION_OPTIONS.index(default_resolution) if default_resolution in RESOLUTION_OPTIONS else 0
        self.resolution_combo.setCurrentIndex(max(default_index, 0))
        self.bit_depth_combo = FocusWheelComboBox(
            self.global_input_controls
        )
        for bits in BIT_DEPTH_OPTIONS:
            self.bit_depth_combo.addItem(f"{bits}-bit (Max {(1 << bits) - 1})", bits)
        default_bit_index = BIT_DEPTH_OPTIONS.index(DEFAULT_BIT_DEPTH) if DEFAULT_BIT_DEPTH in BIT_DEPTH_OPTIONS else 0
        self.bit_depth_combo.setCurrentIndex(default_bit_index)
        self.output_root = PathRow("folder", dialog_title="Select Output Root Folder")
        self.output_root.setParent(self.global_input_controls)
        self.output_root.setText(str(Path.cwd() / "Thermal_RAW_Results"))
        root.addWidget(settings)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        splitter = self.main_splitter
        root.addWidget(splitter, 1)

        left = QWidget(); left.setObjectName("controlPanel"); left_lay = QVBoxLayout(left)
        self.stack = QStackedWidget()
        left_lay.addWidget(self.stack)
        self.run_btn = QPushButton("Run / Preview")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setMinimumHeight(44)
        self.run_btn.clicked.connect(self.run_current)
        left_lay.addWidget(self.run_btn)
        splitter.addWidget(left)

        right = QWidget(); right.setObjectName("previewPanel"); right_lay = QVBoxLayout(right)
        self.dual_compare_controls = QWidget()
        compare_controls_layout = QHBoxLayout(self.dual_compare_controls)
        compare_controls_layout.setContentsMargins(0, 0, 0, 0)
        compare_controls_layout.addWidget(QLabel("Comparison View"))
        self.dual_compare_mode = QComboBox()
        self.dual_compare_mode.addItems(["Side by Side", "Swipe Compare"])
        compare_controls_layout.addWidget(self.dual_compare_mode)
        compare_controls_layout.addStretch()
        self.dual_compare_controls.hide()
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.setHandleWidth(6)
        self.right_splitter.setChildrenCollapsible(True)
        preview_area = QWidget()
        previews = QHBoxLayout(preview_area)
        previews.setContentsMargins(0, 0, 0, 0)
        previews.setSpacing(8)
        self.preview_original = PreviewLabel("Original / Input Preview")
        self.preview_result = PreviewLabel("Result Preview")
        self.preview_original.double_clicked.connect(
            lambda: self.open_preview_window(self.preview_original)
        )
        self.preview_result.double_clicked.connect(
            lambda: self.open_preview_window(self.preview_result)
        )
        self.image_preview_windows = {}
        for preview in (self.preview_original, self.preview_result):
            preview.image_changed.connect(
                lambda preview=preview: self.update_image_preview_window(preview)
            )
        self.preview_original_container = QWidget()
        original_preview_layout = QVBoxLayout(self.preview_original_container)
        original_preview_layout.setContentsMargins(0, 0, 0, 0)
        original_preview_layout.setSpacing(6)
        self.preview_original_caption = QLabel("Original / Input Preview")
        self.preview_original_caption.setAlignment(Qt.AlignCenter)
        self.preview_original_caption.setStyleSheet("font-size:16px; font-weight:700; color:#c9e2f2; background:#2b333b; border-left:4px solid #79afd3; border-radius:6px; padding:7px;")
        original_preview_layout.addWidget(self.preview_original_caption)
        original_preview_layout.addWidget(self.preview_original, 1)
        self.preview_result_container = QWidget()
        result_preview_layout = QVBoxLayout(self.preview_result_container)
        result_preview_layout.setContentsMargins(0, 0, 0, 0)
        result_preview_layout.setSpacing(6)
        self.preview_result_caption = QLabel("Result Preview")
        self.preview_result_caption.setAlignment(Qt.AlignCenter)
        self.preview_result_caption.setStyleSheet("font-size:16px; font-weight:700; color:#d4eadc; background:#29372f; border-left:4px solid #6aaa7d; border-radius:6px; padding:7px;")
        result_preview_layout.addWidget(self.preview_result_caption)
        result_preview_layout.addWidget(self.preview_result, 1)
        previews.addWidget(self.preview_original_container, 1)
        previews.addWidget(self.preview_result_container, 1)
        self.dual_swipe_preview = SwipeCompareWidget()
        previews.addWidget(self.dual_swipe_preview, 1)
        self.dual_swipe_preview.hide()
        self.dual_swipe_preview.double_clicked.connect(
            lambda: self.open_preview_window(self.preview_original)
        )
        self.dual_swipe_preview.position_changed.connect(self.sync_dual_swipe_position)
        self.dual_compare_mode.currentIndexChanged.connect(self.update_dual_compare_mode)
        self.right_splitter.addWidget(preview_area)

        self.info_area = QWidget()
        self.info_area.setObjectName("infoPanel")
        info_layout = QVBoxLayout(self.info_area)
        info_layout.setContentsMargins(10, 8, 10, 10)
        info_layout.setSpacing(6)
        info_header = QHBoxLayout()
        self.info_title = QLabel("Information / Log")
        self.info_title.setStyleSheet("font-size:16px; font-weight:700; color:#d7dde5; padding:2px 4px;")
        self.settings_button = QPushButton("Setting")
        self.settings_button.setToolTip(
            "Configure appearance, input, camera mirror, and histogram"
        )
        self.settings_button.clicked.connect(self.show_toolkit_settings)
        info_header.addWidget(self.info_title)
        info_header.addStretch()
        info_layout.addLayout(info_header)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(120)
        self.update_log_font_size(self.app_settings["log_font_size"])
        info_layout.addWidget(self.log, 1)
        self.right_splitter.addWidget(self.info_area)
        self.right_splitter.setCollapsible(0, False)
        self.right_splitter.setCollapsible(1, True)
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.setSizes([650, 50])
        self.right_splitter.handle(1).setToolTip(
            "Drag down to resize or fully hide Information / Log"
        )
        right_lay.addWidget(self.right_splitter, 1)
        action_row = QHBoxLayout()
        self.save_current_btn = QPushButton("Save Current Preview PNG")
        self.save_current_btn.setObjectName("saveButton")
        self.save_current_btn.clicked.connect(self.save_current_preview)
        self.open_result_btn = QPushButton("Open Last Result Folder")
        self.open_result_btn.setObjectName("openButton")
        self.open_result_btn.clicked.connect(self.open_last_result)
        action_row.addWidget(self.save_current_btn)
        action_row.addWidget(self.open_result_btn)
        action_row.addStretch()
        action_row.addWidget(self.settings_button)
        right_lay.addLayout(action_row)
        splitter.addWidget(right)
        splitter.setSizes([350, 1050])

        self._create_pages()
        self.change_page(self.FUNCTIONS.index("Image Algorithm Compare"))

    def category_name_for_function(self, function_name):
        return next(
            category_name
            for category_name, function_names in self.FUNCTION_GROUPS
            if function_name in function_names
        )

    def select_function_from_menu(self, function_index):
        function_index = int(function_index)
        if function_index == self.current_function_index():
            self.function_actions[function_index].setChecked(True)
            return
        self.change_page(function_index)

    def current_function_index(self):
        return int(self.active_function_index)

    def change_page(self, function_index):
        for window in self.image_preview_windows.values():
            window.hide()
            window.clear_frame()
        self.close_uvc_video_window()
        self.close_uvc_histogram_window()
        self.close_raw_histogram_window()
        if function_index is None:
            return
        index = int(function_index)
        if not 0 <= index < len(self.FUNCTIONS):
            return
        self.active_function_index = index
        function_name = self.FUNCTIONS[index]
        category_name = self.category_name_for_function(function_name)
        self.function_actions[index].setChecked(True)
        single_raw_page_index = self.FUNCTIONS.index("Single Image Statistics")
        viewer_uvc_page_index = self.FUNCTIONS.index("Single UVC Statistic")
        uvc_page_index = len(self.FUNCTIONS) - 1
        raw_algorithm_page_index = len(self.FUNCTIONS) - 2
        dual_uvc_page_index = len(self.FUNCTIONS) - 3
        if index != viewer_uvc_page_index and self.viewer_uvc_worker is not None:
            self.viewer_uvc_restart_when_stopped = False
            self.stop_uvc_viewer(log_message=False)
        if index != uvc_page_index and (
            self.uvc_worker is not None or self.uvc_processing_worker is not None
        ):
            self.uvc_restart_when_stopped = False
            self.stop_uvc_camera(log_message=False)
        if index != dual_uvc_page_index and (
            self.dual_uvc_worker_a is not None
            or self.dual_uvc_worker_b is not None
        ):
            self.dual_uvc_restart_when_stopped = False
            self.stop_dual_uvc_cameras(log_message=False)
        self.stack.setCurrentIndex(index)
        self.category_badge.setText(category_name)
        self.function_title.setText(function_name)
        self.function_description.setText(self.FUNCTION_DESCRIPTIONS[index])
        show_comparison, original_title, result_title = self.PREVIEW_LAYOUTS[index]
        if index == raw_algorithm_page_index:
            result_title = self.raw_preview_stage_combo.currentText()
        elif index == uvc_page_index:
            result_title = self.uvc_result_title()
        elif index == dual_uvc_page_index:
            original_title, result_title = self.dual_uvc_image_names()
        elif function_name == "Dual Image Compare":
            original_title, result_title = self.dual_image_names()
        self.preview_original_container.setVisible(show_comparison)
        self.preview_original.setVisible(show_comparison)
        self.preview_original.set_title(original_title)
        self.preview_original_caption.setText(original_title)
        self.preview_result.set_title(result_title)
        self.preview_result_caption.setText(result_title)
        self.update_image_preview_titles()
        self.preview_original.set_array(None)
        self.preview_result.set_array(None)
        self.update_dual_compare_mode()
        action_labels = {
            "Single Image Statistics": "Save Analysis Results",
            "Average RAW Folder": "Average and Save",
            "NUC Calculate": "Calculate and Save NUC",
            "Bad pixel calibration": "Run / Generate dpc_table.bin",
            "NUC_Analyze": "Save",
            "High / Low Analyzer": "Analyze and Save",
            "Temporal Drift Analyzer": "Save",
            "BPR Bad Pixel Marking": "Save",
            "Dual Image Compare": "Save Comparison Results",
            "Single UVC Statistic": "Capture and Save Analysis",
            "Dual UVC Compare": "Start Cameras",
            "Image Algorithm Compare": "Save Analysis Results",
            "UVC Algorithm Compare": "Start Camera",
        }
        save_only = function_name in (
            "Single Image Statistics", "Dual Image Compare", "Image Algorithm Compare",
            "Temporal Drift Analyzer",
        )
        run_button_object_name = "primaryButton"
        self.run_btn.setText(action_labels[function_name])
        self.run_btn.setToolTip(
            "The preview updates automatically. Save images and analysis reports to a new result folder."
            if save_only else action_labels[function_name]
        )
        if self.run_btn.objectName() != run_button_object_name:
            self.run_btn.setObjectName(run_button_object_name)
            self.run_btn.style().unpolish(self.run_btn)
            self.run_btn.style().polish(self.run_btn)
        self.run_btn.setVisible(
            index not in (
                dual_uvc_page_index,
                uvc_page_index,
            )
        )
        if index == viewer_uvc_page_index:
            self.run_btn.setEnabled(
                self.viewer_uvc_worker is not None
                and self.viewer_uvc_frame is not None
            )
        elif function_name == "Single Image Statistics":
            self.run_btn.setEnabled(getattr(self, "_single_image_save_ready", False))
        elif function_name == "Dual Image Compare":
            self.run_btn.setEnabled(getattr(self, "_dual_image_save_ready", False))
        elif function_name == "Temporal Drift Analyzer":
            self.run_btn.setEnabled(self._drift_result is not None)
        else:
            self.run_btn.setEnabled(
                index != raw_algorithm_page_index
                or (
                    self.raw_algorithm_source is not None
                    )
            )
        self.save_current_btn.setVisible(
            index not in (3, viewer_uvc_page_index, dual_uvc_page_index)
            and function_name not in ("Temporal Drift Analyzer", "Bad pixel calibration")
        )
        self.log.clear()
        self.log.append(
            f"Selected: {category_name} / {function_name}"
        )
        if (
            index == raw_algorithm_page_index
            and self.raw_algorithm_source is not None
        ):
            self.refresh_raw_pipeline_preview()

        if function_name == "NUC_Analyze":
            self.auto_preview_nuc()
        if function_name == "Temporal Drift Analyzer":
            if self._drift_result is not None and self._drift_result_source == self.drift_source_key():
                self.show_drift_result()
            elif self._drift_worker is None and not self.drift_preview_timer.isActive():
                self.schedule_drift_preview()
        if function_name == "BPR Bad Pixel Marking":
            self.auto_preview_bpr()

    @property
    def width(self): return self.resolution_combo.currentData()[0]

    @property
    def height(self): return self.resolution_combo.currentData()[1]

    @property
    def bit_depth(self): return self.bit_depth_combo.currentData()

    @property
    def raw_max(self): return (1 << self.bit_depth) - 1

    def log_stats(self, name, image, stats=None):
        s = stats if stats is not None else image_stats(image)
        self.log.append(f"\n[{name}]\nMin={s['min']:.3f}  Max={s['max']:.3f}  Mean={s['mean']:.3f}  Median={s['median']:.3f}  Std={s['std']:.3f}")

    def make_output(self, name):
        out = unique_result_dir(self.output_root.text(), name)
        self.last_output_dir = str(out)
        self.log.append(f"\nOutput: {out}")
        return out

    def run_current(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            idx = self.current_function_index()
            handlers = [
                self.run_viewer,
                self.run_average,
                self.run_nuc_calculate,
                self.run_nuc_analyze,
                self.run_high_low,
                self.run_drift,
                self.run_bad_pixel_calibration,
                self.run_bpr_marking,
                self.run_raw_quality_compare,
                self.capture_uvc_viewer,
                self.start_dual_uvc_cameras,
                self.run_raw_algorithm,
                self.start_uvc_camera,
            ]
            handlers[idx]()
        except Exception as exc:
            self.log.append(f"\nERROR: {exc}")
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def open_last_result(self):
        if not self.last_output_dir or not os.path.isdir(self.last_output_dir):
            QMessageBox.information(self,"No Result","No result folder yet."); return
        path=self.last_output_dir
        if sys.platform.startswith("win"): os.startfile(path)
        elif sys.platform=="darwin": os.system(f'open "{path}"')
        else: os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')

    def closeEvent(self, event):
        self._drift_closing = True
        self.drift_preview_timer.stop()
        if self._drift_worker is not None:
            self._drift_worker.requestInterruption()
            if not self._drift_worker.wait(100):
                QTimer.singleShot(200, self.close)
                event.ignore()
                return
        self.uvc_restart_when_stopped = False
        self.dual_uvc_restart_when_stopped = False
        self.viewer_uvc_restart_when_stopped = False
        self.close_uvc_video_window()
        self.close_uvc_histogram_window()
        self.close_raw_histogram_window()
        self.camera_refresh_timer.stop()
        self.uvc_watchdog_timer.stop()
        self.viewer_uvc_display_timer.stop()
        self.dual_uvc_display_timer.stop()
        if not self.stop_uvc_viewer(
            log_message=False, wait_ms=3000
        ):
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        if not self.stop_dual_uvc_recording(
            reason="Application closing", wait_ms=3000
        ):
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        dual_workers = (
            self.dual_uvc_worker_a,
            self.dual_uvc_worker_b,
        )
        if any(
            worker is not None and worker.isRunning()
            for worker in dual_workers
        ):
            if not self.dual_uvc_close_pending:
                self.dual_uvc_close_pending = True
                self.dual_uvc_start_btn.setEnabled(False)
                self.dual_uvc_stop_btn.setEnabled(False)
                self.dual_uvc_capture_btn.setEnabled(False)
                self.dual_uvc_record_start_btn.setEnabled(False)
                self.dual_uvc_record_stop_btn.setEnabled(False)
                self.dual_uvc_status_a.setText(
                    f"Stopping {self.dual_uvc_image_name_a.currentText()} before exit..."
                )
                self.dual_uvc_status_b.setText(
                    f"Stopping {self.dual_uvc_image_name_b.currentText()} before exit..."
                )
                for dual_worker in dual_workers:
                    if dual_worker is not None:
                        dual_worker.stop()
                QTimer.singleShot(
                    UVC_CLOSE_TIMEOUT_MS,
                    self.force_close_after_dual_uvc_timeout,
                )
            event.ignore()
            return
        self.dual_uvc_worker_a = None
        self.dual_uvc_worker_b = None
        if not self.stop_uvc_recording(
            reason="Application closing", wait_ms=3000
        ):
            self.uvc_status.setText("Finishing recording before exit...")
            QTimer.singleShot(500, self.close)
            event.ignore()
            return
        worker = self.uvc_worker
        if worker is not None and worker.isRunning():
            if not self.uvc_close_pending:
                self.uvc_close_pending = True
                self.uvc_start_btn.setEnabled(False)
                self.uvc_stop_btn.setEnabled(False)
                self.uvc_capture_btn.setEnabled(False)
                self.uvc_record_start_btn.setEnabled(False)
                self.uvc_record_stop_btn.setEnabled(False)
                self.uvc_status.setText("Stopping camera before exit...")
                worker.stop()
                QTimer.singleShot(
                    UVC_CLOSE_TIMEOUT_MS,
                    self.force_close_after_uvc_timeout,
                )
            event.ignore()
            return
        self.uvc_worker = None
        self.stop_uvc_processing_worker()
        self.uvc_close_pending = False
        super().closeEvent(event)
