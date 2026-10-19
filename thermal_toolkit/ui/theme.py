"""ui / theme."""

from thermal_toolkit.core.qt import QFont


class ThemeMixin:
    """ui / theme behavior composed by the main window."""

    def _apply_ui_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #181a1e;
            }
            QMessageBox {
                background-color: #22252a;
                color: #e7e9ed;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QMessageBox QLabel#qt_msgbox_label {
                color: #e7e9ed;
                background: transparent;
                padding: 8px 10px;
                min-width: 300px;
            }
            QMessageBox QLabel#qt_msgboxex_icon_label {
                background: transparent;
                padding: 8px 4px 8px 10px;
            }
            QMessageBox QDialogButtonBox {
                background: transparent;
            }
            QMessageBox QPushButton {
                background-color: #3a4048;
                color: #f5f6f7;
                border: 1px solid #59636e;
                border-radius: 7px;
                min-width: 84px;
                min-height: 28px;
                padding: 5px 14px;
                font-weight: 600;
            }
            QMessageBox QPushButton:hover {
                background-color: #4a515b;
                border-color: #74808c;
            }
            QMessageBox QPushButton:pressed {
                background-color: #2d3239;
            }
            QMessageBox QPushButton:default {
                background-color: #3f7eae;
                border-color: #69a5d0;
                color: #ffffff;
            }
            QMessageBox QPushButton:default:hover {
                background-color: #4d90c1;
            }
            QDialog#toolkitSettingsDialog {
                background-color: #181a1e;
                color: #e7e9ed;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QDialog#toolkitSettingsDialog QTabWidget::pane {
                background-color: #22252a;
                border: 1px solid #424850;
                border-radius: 7px;
                top: -1px;
            }
            QDialog#toolkitSettingsDialog QTabBar::tab {
                background-color: #292d33;
                color: #bdc5ce;
                border: 1px solid #424850;
                padding: 8px 15px;
                min-width: 110px;
            }
            QDialog#toolkitSettingsDialog QTabBar::tab:selected {
                background-color: #31546c;
                color: #ffffff;
                border-color: #5685a3;
            }
            QDialog#toolkitSettingsDialog QGroupBox#settingSection {
                background-color: #20252b;
                border: 1px solid #46515c;
                border-radius: 8px;
                margin-top: 14px;
                padding: 12px 8px 8px 8px;
                color: #c9e2f2;
                font-weight: 700;
            }
            QDialog#toolkitSettingsDialog QGroupBox#settingSection::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #c9e2f2;
            }
            QWidget#centralWidget {
                background-color: #181a1e;
                color: #e7e9ed;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QWidget#controlPanel, QWidget#previewPanel {
                background-color: #22252a;
                border: 1px solid #3a3f47;
                border-radius: 10px;
            }
            QWidget#infoPanel {
                background-color: #202328;
                border: 1px solid #424850;
                border-radius: 8px;
            }
            QGroupBox#settingsCard {
                background-color: #22252a;
                border: 1px solid #3a3f47;
                border-radius: 10px;
            }
            QGroupBox#dualUvcCameraGroup {
                background-color: #22252a;
                border: 1px solid #3a3f47;
                border-radius: 10px;
            }
            QGroupBox#dualUvcCameraGroup::title {
                subcontrol-origin: padding;
                subcontrol-position: top left;
                left: 12px;
                top: 7px;
                padding: 0;
            }
            QGroupBox#viewerStatsCard {
                background-color: #20252b;
                border: 1px solid #46515c;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
                font-weight: 700;
                color: #c9e2f2;
            }
            QGroupBox#viewerStatsCard::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QGroupBox#pipelineParameterCard {
                background-color: #202328;
                border: 1px solid #424850;
                border-radius: 7px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
                color: #c9e2f2;
                font-weight: 700;
            }
            QGroupBox#pipelineParameterCard::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #c9e2f2;
            }
            QGroupBox[collapsibleGroup="true"]::title {
                font-family: "Segoe UI";
                font-size: 16px;
                font-weight: 700;
                color: #c9e2f2;
            }
            QLabel[viewerStatValue="true"] {
                color: #ffffff;
                background-color: #191c21;
                border: 1px solid #3f4852;
                border-radius: 5px;
                padding: 5px 7px;
                font-family: "Cascadia Mono", "Consolas";
                font-weight: 700;
            }
            QLabel {
                color: #e0e3e8;
                background: transparent;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QListWidget {
                background-color: #191c21;
                color: #f0f1f3;
                border: 1px solid #454b54;
                border-radius: 6px;
                padding: 6px 8px;
                selection-background-color: #4f8fbd;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
            QTextEdit:focus, QListWidget:focus {
                border: 1px solid #86bce2;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #25292f;
                color: #f0f1f3;
                border: 1px solid #4a515b;
                selection-background-color: #4f8fbd;
                padding: 4px;
            }
            QPushButton#categoryBadge {
                background-color: #376f8d;
                color: #f4fbff;
                border: 1px solid #65a4c5;
                border-radius: 7px;
                padding: 7px 24px 7px 22px;
                font-family: "Segoe UI";
                font-size: 20px;
                font-weight: normal;
            }
            QPushButton#categoryBadge:hover,
            QPushButton#categoryBadge:focus {
                background-color: #43819f;
                border-color: #8bc8e7;
            }
            QPushButton#categoryBadge:pressed {
                background-color: #2f607a;
                border-color: #5795b5;
            }
            QPushButton#categoryBadge::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                right: 9px;
            }
            QMenu {
                background-color: #25292f;
                color: #f0f1f3;
                border: 1px solid #555d68;
                border-radius: 6px;
                padding: 5px;
                font-family: "Segoe UI";
                font-size: 17px;
                font-weight: 700;
            }
            QMenu::item {
                border-radius: 4px;
                padding: 7px 30px 7px 12px;
            }
            QMenu::item:selected {
                background-color: #43819f;
                color: #ffffff;
            }
            QMenu::item:checked {
                color: #55b5ff;
                font-weight: 700;
            }
            QMenu::item:checked:selected {
                background-color: #19456b;
                color: #55b5ff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #454b54;
                margin: 4px 8px;
            }
            QListWidget::item {
                color: #e7e9ed;
                padding: 4px 5px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #4f8fbd;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a4048;
                color: #f5f6f7;
                border: 1px solid #555d68;
                border-radius: 7px;
                padding: 7px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4a515b;
                border-color: #69727e;
            }
            QPushButton:pressed {
                background-color: #2d3239;
            }
            QPushButton:disabled {
                background-color: #292d33;
                color: #737a84;
                border-color: #383d44;
            }
            QPushButton#primaryButton {
                background-color: #3f7eae;
                border-color: #69a5d0;
            }
            QPushButton#primaryButton:hover {
                background-color: #4d90c1;
            }
            QPushButton#startButton, QPushButton#saveButton {
                background-color: #397851;
                border-color: #5ca476;
            }
            QPushButton#startButton:hover, QPushButton#saveButton:hover {
                background-color: #458c60;
            }
            QPushButton#stopButton {
                background-color: #984646;
                border-color: #c66a6a;
            }
            QPushButton#stopButton:hover {
                background-color: #ad5050;
            }
            QPushButton#captureButton {
                background-color: #70518c;
                border-color: #9874b5;
            }
            QPushButton#captureButton:hover {
                background-color: #80609d;
            }
            QPushButton#resetButton {
                background-color: #a96f2e;
                border-color: #d09a58;
                color: #fff8ef;
            }
            QPushButton#resetButton:hover {
                background-color: #bc7e35;
            }
            QPushButton#startButton:disabled, QPushButton#stopButton:disabled,
            QPushButton#captureButton:disabled, QPushButton#resetButton:disabled,
            QPushButton#saveButton:disabled {
                background-color: #292d33;
                color: #737a84;
                border-color: #383d44;
            }
            QPushButton[buttonRole="browse"] {
                background-color: #376f8d;
                border-color: #65a4c5;
                min-width: 66px;
            }
            QPushButton[buttonRole="browse"]:hover {
                background-color: #43819f;
            }
            QCheckBox {
                color: #e7e9ed;
                spacing: 8px;
                font-size: 14px;
                font-weight: 600;
            }
            QCheckBox#qualityScoreToggle {
                color: #edf4f8;
                font-family: "Segoe UI";
                font-size: 20px;
                font-weight: 600;
                spacing: 0px;
                padding: 0px;
            }
            QScrollArea#formScroll,
            QWidget#formViewport,
            QWidget#formContent {
                background-color: #22252a;
                color: #e7e9ed;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #191c21;
                width: 12px;
                margin: 0;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #4a515b;
                min-height: 28px;
                border-radius: 5px;
            }
            QScrollBar:horizontal {
                background-color: #191c21;
                height: 12px;
                margin: 0;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background-color: #4a515b;
                min-width: 28px;
                border-radius: 5px;
            }
            QScrollBar::handle:hover {
                background-color: #626c78;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0;
                height: 0;
                border: none;
                background: transparent;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: transparent;
            }
            QTextEdit {
                font-family: "Cascadia Mono", "Consolas";
                font-size: 17px;
            }
            QSplitter::handle {
                background: transparent;
            }
            QSplitter::handle:horizontal {
                width: 6px;
            }
            QSplitter::handle:vertical {
                height: 6px;
            }
            QToolTip {
                background-color: #30343a;
                color: #f5f6f7;
                border: 1px solid #555d68;
                padding: 5px;
            }
        """)

    def _apply_category_badge_font(self):
        """Match the adjacent title font and size while keeping regular weight."""
        self.function_title.ensurePolished()
        font = QFont(self.function_title.font())
        font.setWeight(QFont.Normal)
        font.setBold(False)
        font.setStretch(QFont.Unstretched)
        self.category_badge.setFont(font)
