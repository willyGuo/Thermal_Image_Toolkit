"""ui / widgets / inputs."""

from thermal_toolkit.core.qt import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
    Qt,
    pyqtSignal,
)


class FocusWheelGuard:
    """Allow wheel changes only after explicit mouse or keyboard focus."""

    def _initialize_wheel_guard(self):
        self.setFocusPolicy(Qt.StrongFocus)
        self._wheel_input_enabled = False

    def mousePressEvent(self, event):
        self._wheel_input_enabled = True
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        if event.reason() in (
            Qt.MouseFocusReason,
            Qt.TabFocusReason,
            Qt.BacktabFocusReason,
        ):
            self._wheel_input_enabled = True
        elif event.reason() != Qt.PopupFocusReason:
            self._wheel_input_enabled = False
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        # Keep the marker while a popup or child editor temporarily owns focus.
        # wheelEvent() still rejects input whenever hasFocus() is false;
        # the next focusInEvent decides whether wheel input is enabled again.
        super().focusOutEvent(event)

    def wheelEvent(self, event):
        if not self.hasFocus() or not self._wheel_input_enabled:
            event.ignore()
            return
        super().wheelEvent(event)


class FocusWheelComboBox(FocusWheelGuard, QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._initialize_wheel_guard()


class FocusWheelSpinBox(FocusWheelGuard, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Commit typed numbers on Enter/focus loss, not after each digit.
        self.setKeyboardTracking(False)
        self._initialize_wheel_guard()


class FocusWheelDoubleSpinBox(FocusWheelGuard, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setKeyboardTracking(False)
        self._initialize_wheel_guard()


class PathRow(QWidget):
    path_selected = pyqtSignal(str)

    def __init__(self, mode="file", filter_text="All Files (*)", dialog_title=None):
        super().__init__()
        self.mode = mode
        self.filter_text = filter_text
        self.dialog_title = dialog_title or ("Select Folder" if mode == "folder" else "Select File")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(self.dialog_title)
        self.btn = QPushButton("Browse")
        self.btn.setProperty("buttonRole", "browse")
        self.btn.setToolTip(self.dialog_title)
        self.btn.clicked.connect(self.browse)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)

    def browse(self):
        if self.mode == "folder":
            p = QFileDialog.getExistingDirectory(self, self.dialog_title)
        else:
            p, _ = QFileDialog.getOpenFileName(self, self.dialog_title, "", self.filter_text)
        if p:
            self.edit.setText(p)
            self.path_selected.emit(p)

    def text(self):
        return self.edit.text().strip()

    def setText(self, text):
        self.edit.setText(text)
