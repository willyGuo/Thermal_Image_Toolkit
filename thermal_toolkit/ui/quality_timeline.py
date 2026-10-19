"""ui / quality_timeline."""

from collections import deque
import numpy as np
import time
from thermal_toolkit.core.constants import QUALITY_TIMELINE_METRICS
from thermal_toolkit.core.qt import (
    QColor,
    QDialog,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPointF,
    QRectF,
    QVBoxLayout,
    QWidget,
    Qt,
    pyqtSignal,
)


class QualityScoreTimelineChart(QWidget):
    open_requested = pyqtSignal()
    data_changed = pyqtSignal()

    def __init__(self, series, history_seconds=60.0, parent=None):
        super().__init__(parent)
        self.series = tuple(series)
        self.history_seconds = float(history_seconds)
        self.metric_name = "overall_score"
        self.metric_display_name = "Total Score"
        self.samples = deque(maxlen=1200)
        self.paused = False
        self.setMinimumSize(360, 210)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Double-click to open a resizable quality-score timeline")

    def set_display_metric(self, metric_name):
        metric_map = {
            metric: display for display, metric in QUALITY_TIMELINE_METRICS
        }
        if metric_name not in metric_map:
            metric_name = "overall_score"
        self.metric_name = metric_name
        self.metric_display_name = metric_map[metric_name]
        self.update()
        self.data_changed.emit()

    def set_history_seconds(self, seconds):
        self.history_seconds = float(max(10, min(600, int(seconds))))
        self.update()
        self.data_changed.emit()

    def set_series_labels(self, **labels):
        self.series = tuple(
            (key, str(labels.get(key, label)), color)
            for key, label, color in self.series
        )
        self.update()
        self.data_changed.emit()

    def append_metrics(self, timestamp=None, **stream_metrics):
        if self.paused:
            return
        timestamp = time.perf_counter() if timestamp is None else float(timestamp)
        values = {}
        for key, _label, _color in self.series:
            metrics = stream_metrics.get(key)
            values[key] = None if metrics is None else dict(metrics)
        self.samples.append((timestamp, values))
        cutoff = timestamp - self.history_seconds * 1.25
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        self.update()
        self.data_changed.emit()

    def append_scores(self, timestamp=None, **scores):
        self.append_metrics(
            timestamp,
            **{
                key: {self.metric_name: value}
                for key, value in scores.items()
            },
        )

    def set_paused(self, paused):
        paused = bool(paused)
        if paused != self.paused and self.samples:
            self.samples.append((time.perf_counter(), {}))
        self.paused = paused
        self.update()
        self.data_changed.emit()

    def clear(self):
        self.samples.clear()
        self.update()
        self.data_changed.emit()

    def copy_state_from(self, source):
        self.series = tuple(source.series)
        self.samples = deque(source.samples, maxlen=1200)
        self.paused = source.paused
        self.history_seconds = source.history_seconds
        self.metric_name = source.metric_name
        self.metric_display_name = source.metric_display_name
        self.update()

    def excel_metric_definition(self):
        return [(self.metric_display_name, self.metric_name)]

    def _metric_value(self, values, key):
        stream = values.get(key)
        if isinstance(stream, dict):
            return stream.get(self.metric_name)
        return stream

    def excel_timelines(self):
        if not self.samples:
            return {}
        end_time = self.samples[-1][0]
        start_time = end_time - self.history_seconds
        visible_samples = [
            item for item in self.samples if item[0] >= start_time
        ]
        if not visible_samples:
            return {}
        first_time = visible_samples[0][0]
        timelines = {}
        for key, label, _color in self.series:
            timelines[label] = [
                {
                    "elapsed_seconds": round(timestamp - first_time, 3),
                    "metrics": {
                        self.metric_name: self._metric_value(values, key)
                    },
                }
                for timestamp, values in visible_samples
            ]
        return timelines

    def _series_values(self, key, visible_samples):
        return [
            self._metric_value(values, key)
            for _timestamp, values in visible_samples
            if self._metric_value(values, key) is not None
        ]

    @staticmethod
    def _trend_text(values):
        if len(values) < 6:
            return "Collecting"
        recent = values[-10:]
        half = max(1, len(recent) // 2)
        change = float(np.mean(recent[half:]) - np.mean(recent[:half]))
        if change > 1.0:
            return "Improving"
        if change < -1.0:
            return "Declining"
        return "Stable"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#1b1e23"))

        plot = QRectF(48, 42, max(40, self.width() - 66), max(40, self.height() - 76))
        painter.setPen(QPen(QColor("#39414a"), 1))
        painter.setBrush(QColor("#20252b"))
        painter.drawRoundedRect(plot, 5, 5)

        label_font = QFont("Segoe UI")
        label_font.setPixelSize(11)
        painter.setFont(label_font)
        for score in range(0, 101, 20):
            y = plot.bottom() - (score / 100.0) * plot.height()
            painter.setPen(QPen(QColor("#3b444d"), 1, Qt.DotLine))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#aeb8c2"))
            painter.drawText(QRectF(4, y - 9, 39, 18), Qt.AlignRight | Qt.AlignVCenter, str(score))

        visible_samples = list(self.samples)
        if visible_samples:
            end_time = visible_samples[-1][0]
        else:
            end_time = time.perf_counter()
        start_time = end_time - self.history_seconds
        visible_samples = [item for item in visible_samples if item[0] >= start_time]

        for seconds_ago in (int(self.history_seconds), int(self.history_seconds / 2), 0):
            x = plot.right() - (seconds_ago / self.history_seconds) * plot.width()
            painter.setPen(QColor("#8f9aa5"))
            text = "Now" if seconds_ago == 0 else f"-{seconds_ago}s"
            painter.drawText(QRectF(x - 24, plot.bottom() + 5, 48, 18), Qt.AlignCenter, text)

        legend_x = plot.left()
        for key, label, color_name in self.series:
            color = QColor(color_name)
            values = self._series_values(key, visible_samples)
            latest = "--" if not values else f"{values[-1]:.1f}"
            average = "--" if not values else f"{float(np.mean(values)):.1f}"
            legend = f"{label}  {latest}  Avg {average}  {self._trend_text(values)}"
            painter.setPen(QPen(color, 3))
            painter.drawLine(QPointF(legend_x, 23), QPointF(legend_x + 18, 23))
            painter.setPen(QColor("#e5edf3"))
            painter.drawText(QRectF(legend_x + 24, 10, 250, 26), Qt.AlignLeft | Qt.AlignVCenter, legend)
            legend_x += min(285, max(190, self.width() // max(1, len(self.series))))

            path = QPainterPath()
            path_active = False
            for timestamp, score_values in visible_samples:
                value = self._metric_value(score_values, key)
                if value is None:
                    path_active = False
                    continue
                x = plot.left() + ((timestamp - start_time) / self.history_seconds) * plot.width()
                y = plot.bottom() - (max(0.0, min(100.0, value)) / 100.0) * plot.height()
                if not path_active:
                    path.moveTo(x, y)
                    path_active = True
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(color, 2.3))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

        if not visible_samples:
            painter.setPen(QColor("#84909b"))
            painter.drawText(
                plot,
                Qt.AlignCenter,
                f"Waiting for {self.metric_display_name}...",
            )
        if self.paused:
            painter.fillRect(plot, QColor(20, 23, 27, 145))
            paused_font = QFont("Segoe UI")
            paused_font.setPixelSize(17)
            paused_font.setBold(True)
            painter.setFont(paused_font)
            painter.setPen(QColor("#d5dde4"))
            painter.drawText(plot, Qt.AlignCenter, "Quality scoring paused")

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_requested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class QualityTimelineDialog(QDialog):
    def __init__(self, source_chart, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self.resize(1050, 620)
        self.setMinimumSize(680, 420)
        layout = QVBoxLayout(self)
        self.chart = QualityScoreTimelineChart(
            source_chart.series, source_chart.history_seconds, self
        )
        layout.addWidget(self.chart)
        self.source_chart = source_chart
        source_chart.data_changed.connect(self.sync_from_source)
        self.sync_from_source()

    def sync_from_source(self):
        self.chart.copy_state_from(self.source_chart)
