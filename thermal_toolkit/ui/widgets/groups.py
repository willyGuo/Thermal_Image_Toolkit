"""ui / widgets / groups."""

from thermal_toolkit.core.qt import (
    QCheckBox,
    QColor,
    QFont,
    QGroupBox,
    QLabel,
    QPainter,
    QPen,
    QRect,
    QSize,
    QStyle,
    QStyleOptionGroupBox,
    Qt,
    pyqtSignal,
)


class QualityHelpTitle(QLabel):
    help_requested = pyqtSignal()

    def __init__(self, title, parent=None):
        super().__init__(f"{title}  ⓘ", parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to view score definitions and formulas")
        self.setAccessibleName(f"{title} help")
        self._set_hovered(False)

    def _set_hovered(self, hovered):
        if hovered:
            self.setStyleSheet(
                "color:#86c8f4; font-size:18px; font-weight:800; "
                "padding:2px 4px;"
            )
        else:
            self.setStyleSheet(
                "color:#c9e2f2; font-size:18px; font-weight:600; "
                "padding:2px 4px;"
            )

    def enterEvent(self, event):
        self._set_hovered(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_hovered(False)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.help_requested.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CollapsibleGroupBox(QGroupBox):
    """QGroupBox whose title toggles the visibility of its body."""

    expanded_changed = pyqtSignal(bool)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._section_title = str(title)
        self._expanded = True
        self._stored_visibility = {}
        self.setProperty("collapsibleGroup", True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._update_title()

    def _update_title(self):
        marker = "-" if self._expanded else "+"
        super().setTitle(f"{marker}  {self._section_title}")
        self.setToolTip(
            "Click the title to collapse" if self._expanded
            else "Click the title to expand"
        )

    def setTitle(self, title):
        if not hasattr(self, "_section_title"):
            super().setTitle(title)
            return
        self._section_title = str(title)
        self._update_title()

    def _title_rect(self):
        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QStyle.CC_GroupBox,
            option,
            QStyle.SC_GroupBoxLabel,
            self,
        )

    @staticmethod
    def _layout_widgets(layout):
        widgets = []
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget()
            if widget is not None:
                widgets.append(widget)
                continue
            child_layout = item.layout()
            if child_layout is not None:
                widgets.extend(CollapsibleGroupBox._layout_widgets(child_layout))
        return widgets

    def set_body_widget_visible(self, widget, visible):
        """Remember conditional rows even while the section is collapsed."""
        if self._expanded:
            widget.setVisible(visible)
        else:
            self._stored_visibility[widget] = bool(visible)
            widget.hide()

    def _header_rect(self):
        return QRect(0, 0, self.width(), max(28, self._title_rect().bottom() + 10))

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        layout = self.layout()
        if layout is None:
            return

        if not expanded:
            self._stored_visibility = {}
            for widget in self._layout_widgets(layout):
                self._stored_visibility[widget] = not widget.isHidden()
                widget.hide()
        else:
            for widget, was_visible in self._stored_visibility.items():
                widget.setVisible(was_visible)
            self._stored_visibility.clear()

        self._expanded = expanded
        self._update_title()
        self.setMaximumHeight(
            16777215 if expanded else self._title_rect().bottom() + 10
        )
        self.updateGeometry()
        self.expanded_changed.emit(expanded)

    def toggle_expanded(self):
        self.set_expanded(not self._expanded)

    def is_expanded(self):
        return self._expanded

    def mouseMoveEvent(self, event):
        self.setCursor(
            Qt.PointingHandCursor
            if self._header_rect().contains(event.pos())
            else Qt.ArrowCursor
        )
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self._header_rect().contains(event.pos())
        ):
            self.toggle_expanded()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self.hasFocus() and event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.toggle_expanded()
            event.accept()
            return
        super().keyPressEvent(event)


class QualityScoreCheckBox(QCheckBox):
    """Large dark-theme checkbox with an explicit painted check mark."""

    def __init__(self, text="Calculate Quality Score", parent=None):
        super().__init__(text, parent)
        self.setObjectName("qualityScoreToggle")
        self._hovered = False
        font = self.font()
        font.setFamily("Segoe UI")
        font.setPixelSize(20)
        font.setWeight(QFont.DemiBold)
        font.setStretch(QFont.Unstretched)
        self.setFont(font)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(42)
        self.setMinimumWidth(280)

    def sizeHint(self):
        return QSize(300, 42)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        box_size = 22
        box_x = 2
        box_y = (self.height() - box_size) // 2
        box_rect = QRect(box_x, box_y, box_size, box_size)

        if self.isChecked():
            fill_color = QColor("#4f9dcc" if not self._hovered else "#62b2df")
            border_color = QColor("#a9ddf8")
        else:
            fill_color = QColor("#1b1f24" if not self._hovered else "#252b31")
            border_color = QColor("#7e8995" if not self._hovered else "#a9b4bf")
        if not self.isEnabled():
            fill_color = QColor("#2a2d31")
            border_color = QColor("#555b62")

        painter.setPen(QPen(border_color, 1.5))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(box_rect, 5, 5)

        if self.isChecked():
            check_pen = QPen(QColor("#ffffff"), 2.6)
            check_pen.setCapStyle(Qt.RoundCap)
            check_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(check_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(box_x + 5, box_y + 11, box_x + 9, box_y + 15)
            painter.drawLine(box_x + 9, box_y + 15, box_x + 17, box_y + 7)

        text_color = QColor("#edf4f8" if self.isEnabled() else "#7d858d")
        if self._hovered and self.isEnabled():
            text_color = QColor("#ffffff")
        painter.setPen(text_color)
        painter.setFont(self.font())
        metrics = painter.fontMetrics()
        text_y = (self.height() + metrics.ascent() - metrics.descent()) // 2
        painter.drawText(34, text_y, self.text())
