"""ui / widgets / preview."""

import numpy as np
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import (
    QColor,
    QFont,
    QImage,
    QLabel,
    QPainter,
    QPen,
    QPixmap,
    QPointF,
    QRectF,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSignal,
)


class PreviewLabel(QLabel):
    double_clicked = pyqtSignal()
    image_changed = pyqtSignal()

    def __init__(self, title):
        super().__init__()
        self.title = title
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 200)
        self.setStyleSheet("background:#121417; color:#aeb7c2; border:1px solid #424850; border-radius:8px; font-size:15px; font-weight:600;")
        self.setText("Waiting for image...")

    def set_title(self, title):
        self.title = title
        if self._pixmap is None:
            self.setText("Waiting for image...")

    def set_array(self, array):
        if array is None:
            self._pixmap = None
            self.setText("Waiting for image...")
            self.image_changed.emit()
            return
        if array.ndim == 2:
            img8 = np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))
            qimg = QImage(img8.data, img8.shape[1], img8.shape[0], img8.strides[0], QImage.Format_Grayscale8).copy()
        else:
            rgb = np.ascontiguousarray(array[:, :, :3])
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._rescale()
        self.image_changed.emit()

    def _rescale(self):
        if self._pixmap:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class SwipeCompareWidget(QWidget):
    double_clicked = pyqtSignal()
    position_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = (None, None)
        self.names = ("Image A", "Image B")
        self.position = 0.5
        self.dragging = False
        self.setMinimumSize(360, 200)
        self.setMouseTracking(True)

    def set_frames(self, frames, names):
        self.frames = tuple(frames)
        self.names = tuple(names)
        self.update()

    def set_position(self, position):
        self.position = max(0.0, min(1.0, float(position)))
        self.update()

    def image_rect(self):
        if any(frame is None for frame in self.frames):
            return QRectF()
        width = max(frame.width() for frame in self.frames)
        height = max(frame.height() for frame in self.frames)
        scale = min(self.width() / width, self.height() / height)
        return QRectF((self.width() - width * scale) / 2,
                      (self.height() - height * scale) / 2,
                      width * scale, height * scale)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        area = self.image_rect()
        if area.isEmpty():
            painter.setPen(QColor("#aeb7c2"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for both images...")
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        split = area.left() + area.width() * self.position
        clips = (QRectF(area.left(), area.top(), split - area.left(), area.height()),
                 QRectF(split, area.top(), area.right() - split, area.height()))
        for frame, clip in zip(self.frames, clips):
            scale = min(area.width() / frame.width(), area.height() / frame.height())
            target = QRectF(0, 0, frame.width() * scale, frame.height() * scale)
            target.moveCenter(area.center())
            painter.save()
            painter.setClipRect(clip)
            painter.drawPixmap(target, frame, QRectF(frame.rect()))
            painter.restore()
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(QPointF(split, area.top()), QPointF(split, area.bottom()))
        font = QFont("Segoe UI", 11)
        font.setBold(False)
        painter.setFont(font)
        for name, alignment in zip(self.names, (Qt.AlignLeft, Qt.AlignRight)):
            label = QRectF(area.left() + 10, area.top() + 8, max(0, area.width() - 20), 26)
            painter.setPen(QColor("#000000"))
            painter.drawText(label.translated(1, 1), alignment | Qt.AlignVCenter, name)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(label, alignment | Qt.AlignVCenter, name)

    def mousePressEvent(self, event):
        area = self.image_rect()
        split = area.left() + area.width() * self.position
        if (event.button() == Qt.LeftButton and not area.isEmpty()
                and area.top() <= event.y() <= area.bottom()
                and abs(event.x() - split) <= 16):
            self.dragging = True
            self.setCursor(Qt.SplitHCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        area = self.image_rect()
        if not area.isEmpty():
            if self.dragging:
                self.set_position((event.x() - area.left()) / area.width())
                self.position_changed.emit(self.position)
            near_bar = abs(event.x() - area.left() - area.width() * self.position) <= 16
            self.setCursor(Qt.SplitHCursor if self.dragging or near_bar else Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class UvcVideoWindow(QWidget):
    """Resizable image-only window for UVC views and recording layouts."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self._pixmap = None
        self.setWindowTitle("UVC Recording View")
        self.resize(1280, 720)
        self.setMinimumSize(480, 320)
        self.setStyleSheet("background:#000000;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#000000; border:none;")
        layout.addWidget(self.image_label)

    def set_bgr_frame(self, frame):
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            return
        bgr = np.ascontiguousarray(array[:, :, :3].astype(np.uint8))
        bgr_format = getattr(QImage, "Format_BGR888", None)
        if bgr_format is not None:
            image = QImage(
                bgr.data,
                bgr.shape[1],
                bgr.shape[0],
                bgr.strides[0],
                bgr_format,
            ).copy()
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._rescale()

    def clear_frame(self):
        self._pixmap = None
        self.image_label.clear()

    def _rescale(self):
        if self._pixmap is not None and not self._pixmap.isNull():
            self.image_label.setPixmap(
                self._pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)
