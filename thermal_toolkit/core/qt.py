"""PyQt5 imports and optional camera enumeration support."""

try:
    from PyQt5.QtCore import (
        QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
    )
    from PyQt5.QtGui import (
        QColor, QDesktopServices, QFont, QImage, QPainter, QPainterPath,
        QPen, QPixmap
    )
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLabel,
        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
        QTextEdit, QGroupBox, QStackedWidget, QSplitter, QCheckBox,
        QListWidget, QListWidgetItem, QScrollArea, QFrame, QSizePolicy,
        QDialog, QTextBrowser, QDialogButtonBox, QStyle, QStyleOptionGroupBox,
        QTabWidget, QMenu, QAction, QActionGroup, QLayout
    )
except ImportError as exc:
    raise SystemExit("PyQt5 is required. Install with: pip install PyQt5") from exc

try:
    from PyQt5.QtMultimedia import QCameraInfo
except (ImportError, OSError):
    QCameraInfo = None
