"""Dark window chrome with Qt's native system move and resize operations."""

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class WindowButton(QPushButton):
    def __init__(self, action, parent):
        super().__init__(parent)
        self.action = action
        self.setObjectName(action + "WindowButton")
        self.setFixedSize(48, 42)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAccessibleName(action)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#e7e9ed"), 1.2))
        x, y = self.width() // 2 - 5, self.height() // 2 - 5
        if self.action == "minimize":
            painter.drawLine(x, y + 6, x + 10, y + 6)
        elif self.action == "close":
            painter.drawLine(x, y, x + 10, y + 10)
            painter.drawLine(x + 10, y, x, y + 10)
        elif self.window().isMaximized():
            painter.drawRect(x + 3, y, 8, 8)
            painter.fillRect(x, y + 3, 8, 8, QColor("#22252a"))
            painter.drawRect(x, y + 3, 8, 8)
        else:
            painter.drawRect(x, y, 10, 10)


class ToolkitTitleBar(QWidget):
    """Install as the main window's menu widget; leave close handling intact."""

    BORDER = 5

    def __init__(self, window):
        super().__init__(window)
        self.host = window
        self.setObjectName("toolkitTitleBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(44)
        self.setStyleSheet("""
            QWidget#toolkitTitleBar { background: #22252a;
                border-bottom: 1px solid #424850; }
            QLabel { color: #c9e2f2; background: transparent; border: none;
                font-family: 'Segoe UI'; font-size: 17px; font-weight: 600; }
            QPushButton { background: transparent; border: none;
                border-radius: 0; padding: 0; min-width: 0; min-height: 0; }
            QPushButton:hover { background: #3a4048; }
            QPushButton:pressed { background: #31546c; }
            QPushButton#closeWindowButton:hover { background: #c42b1c; }
            QPushButton#closeWindowButton:pressed { background: #a52318; }
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 1)
        layout.setSpacing(0)
        self.title = QLabel(window.windowTitle(), self)
        self.title.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout.addWidget(self.title, 1)
        self.minimize_button = WindowButton("minimize", self)
        self.maximize_button = WindowButton("maximize", self)
        self.close_button = WindowButton("close", self)
        for button in (self.minimize_button, self.maximize_button, self.close_button):
            layout.addWidget(button)
        self.minimize_button.setToolTip("最小化")
        self.close_button.setToolTip("關閉")
        self.minimize_button.clicked.connect(window.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.close_button.clicked.connect(window.close)
        window.windowTitleChanged.connect(self.title.setText)
        window.setWindowFlag(Qt.FramelessWindowHint, True)
        window.setMouseTracking(True)
        window.installEventFilter(self)
        self._sync_state()

    def _sync_state(self):
        border = 0 if self.host.isMaximized() or self.host.isFullScreen() else self.BORDER
        self.host.setContentsMargins(border, border, border, border)
        self.maximize_button.setToolTip("還原" if self.host.isMaximized() else "最大化")
        self.maximize_button.update()
        self.host.unsetCursor()

    def toggle_maximized(self):
        if self.host.isMaximized():
            self.host.showNormal()
        else:
            self.host.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.host.windowHandle():
            self.host.windowHandle().startSystemMove()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximized()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resize_edges(self, point):
        edges = Qt.Edges()
        if self.host.isMaximized() or self.host.isFullScreen():
            return edges
        # The host's width/height attributes store image resolution, not geometry.
        window_size = self.host.size()
        if point.x() < self.BORDER:
            edges |= Qt.LeftEdge
        elif point.x() >= window_size.width() - self.BORDER:
            edges |= Qt.RightEdge
        if point.y() < self.BORDER:
            edges |= Qt.TopEdge
        elif point.y() >= window_size.height() - self.BORDER:
            edges |= Qt.BottomEdge
        return edges

    def eventFilter(self, watched, event):
        if watched is self.host:
            if event.type() == QEvent.WindowStateChange:
                self._sync_state()
            elif event.type() == QEvent.Leave:
                self.host.unsetCursor()
            elif event.type() in (QEvent.MouseMove, QEvent.MouseButtonPress):
                edges = self.resize_edges(event.pos())
                if edges:
                    if edges in (Qt.LeftEdge | Qt.TopEdge, Qt.RightEdge | Qt.BottomEdge):
                        cursor = Qt.SizeFDiagCursor
                    elif edges in (Qt.RightEdge | Qt.TopEdge, Qt.LeftEdge | Qt.BottomEdge):
                        cursor = Qt.SizeBDiagCursor
                    elif edges & (Qt.LeftEdge | Qt.RightEdge):
                        cursor = Qt.SizeHorCursor
                    else:
                        cursor = Qt.SizeVerCursor
                    self.host.setCursor(cursor)
                    if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                        handle = self.host.windowHandle()
                        if handle and handle.startSystemResize(edges):
                            return True
                else:
                    self.host.unsetCursor()
        return super().eventFilter(watched, event)
