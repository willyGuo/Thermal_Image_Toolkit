"""ui / quality_controls."""

import numpy as np
from thermal_toolkit.algorithms.parameters import algorithm_display_name
from thermal_toolkit.core.paths import bundled_resource_path
from thermal_toolkit.core.qt import (
    QDesktopServices,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QUrl,
    QVBoxLayout,
    QWidget,
    Qt,
)
from thermal_toolkit.ui.quality_help import QualityScoreHelpDialog
from thermal_toolkit.ui.quality_timeline import QualityTimelineDialog
from thermal_toolkit.ui.widgets.groups import CollapsibleGroupBox
from thermal_toolkit.ui.image_quality import create_image_quality_group, update_image_quality_labels


class QualityControlsMixin:
    """ui / quality_controls behavior composed by the main window."""

    def _form_page(self):
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("formScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.viewport().setObjectName("formViewport")
        scroll.viewport().setAutoFillBackground(True)
        content = QWidget()
        content.setObjectName("formContent")
        content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        content_layout = QVBoxLayout(content)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        content_layout.addLayout(form)
        content_layout.addStretch()
        scroll.setWidget(content)
        page_layout.addWidget(scroll)
        return page, form

    def create_quality_score_container(
        self,
        title,
        include_temporal=False,
        collapsible=True,
    ):
        group = (
            CollapsibleGroupBox(title)
            if collapsible
            else QGroupBox(title)
        )
        group.setObjectName("viewerStatsCard")
        outer_layout = QVBoxLayout(group)
        outer_layout.setContentsMargins(
            8 if collapsible else 2,
            6 if collapsible else 0,
            8 if collapsible else 2,
            8 if collapsible else 2,
        )
        outer_layout.setSpacing(6)
        layout = QGridLayout()
        outer_layout.addLayout(layout)
        return group, layout

    def create_quality_help_note(self, note_text, include_temporal=False):
        note = QLabel(
            f'{note_text} <a href="quality-help" '
            'style="color:#c9e2f2; text-decoration:none;">'
            '<b>說明 ⓘ</b></a>'
        )
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        note.setTextInteractionFlags(
            Qt.LinksAccessibleByMouse | Qt.LinksAccessibleByKeyboard
        )
        note.setOpenExternalLinks(False)
        note.linkActivated.connect(
            lambda _href, include_temporal=include_temporal:
            self.show_quality_score_help(include_temporal)
        )
        note.setStyleSheet(
            "color:#aeb8c2; font-weight:400; padding-top:4px;"
        )
        return note

    def show_quality_score_help(self, include_temporal=False):
        guide_path = bundled_resource_path(
            "html/quality_score_guide.html" if include_temporal else "html/image_quality_guide.html")
        if guide_path.is_file():
            guide_url = QUrl.fromLocalFile(str(guide_path))
            guide_url.setFragment("uvc" if include_temporal else "raw")
            if QDesktopServices.openUrl(guide_url):
                return

        dialog_key = "uvc" if include_temporal else "raw"
        dialog = self.quality_help_dialogs.get(dialog_key)
        if dialog is None:
            dialog = QualityScoreHelpDialog(include_temporal, self)
            self.quality_help_dialogs[dialog_key] = dialog
        else:
            dialog.reload_guide()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def open_quality_timeline(self, key, source_chart, title):
        title = title.replace(
            "Total Score", source_chart.metric_display_name
        )
        dialog = self.quality_timeline_dialogs.get(key)
        if dialog is None:
            dialog = QualityTimelineDialog(source_chart, title, self)
            self.quality_timeline_dialogs[key] = dialog
        else:
            dialog.sync_from_source()
            dialog.setWindowTitle(title)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @staticmethod
    def create_quality_timeline_group(chart):
        group = CollapsibleGroupBox("Total Score Timeline")
        group.setObjectName("viewerStatsCard")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(chart)
        return group

    def create_quality_score_group(self, title, include_temporal=False):
        if not include_temporal:
            return create_image_quality_group(self, title, paired=True)
        group, layout = self.create_quality_score_container(
            title,
            include_temporal,
        )
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(7)
        comparison_header_labels = []
        for column, header in enumerate(("Metric", "A", "B")):
            header_label = QLabel(header)
            header_label.setStyleSheet(
                "font-weight:700; color:#c9e2f2;"
            )
            header_label.setAlignment(
                Qt.AlignLeft if column == 0 else Qt.AlignRight
            )
            layout.addWidget(header_label, 0, column)
            if column > 0:
                comparison_header_labels.append(header_label)
        group.comparison_header_labels = comparison_header_labels
        fields = [
            ("Total Score", "overall_score"),
            ("Detail Clarity", "detail_clarity_score"),
            ("Contrast Quality", "contrast_quality_score"),
            ("Flat Cleanliness", "flat_cleanliness_score"),
            ("Thermal Uniformity", "thermal_uniformity_score"),
            ("Clipping Control", "clipping_control_score"),
            ("Artifact Control", "artifact_control_score"),
        ]
        if include_temporal:
            fields.extend(
                [
                    ("Flicker Control", "flicker_control_score"),
                    ("Temporal Stability", "temporal_stability_score"),
                ]
            )
        else:
            fields.append(("Quality Confidence", "quality_confidence_score"))
        value_labels = {}
        for row, (display_name, metric_name) in enumerate(fields, start=1):
            metric_label = QLabel(display_name)
            layout.addWidget(metric_label, row, 0)
            metric_labels = []
            for column in range(1, 3):
                value_label = QLabel("--")
                value_label.setProperty("viewerStatValue", True)
                value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value_label.setMinimumWidth(66)
                layout.addWidget(value_label, row, column)
                metric_labels.append(value_label)
            value_labels[metric_name] = metric_labels
        if include_temporal:
            note_text = (
                "UVC Lightweight Quality：使用降採樣、Gaussian/Sobel 與簡化空間評分，"
                "降低即時影像負載。Total Score 權重由 Setting 設定；"
                "時序分數暖機期間顯示 Collecting...。"
            )
        else:
            note_text = (
                "Image Quality：以原始位深完整解析，Flat 區會依灰階層級與連通區域分開統計，"
                "並兼顧典型區域與較差區域；Quality Confidence 表示本張影像可供評分的 Flat/Edge 覆蓋程度。"
            )
        note = self.create_quality_help_note(note_text, include_temporal)
        layout.addWidget(note, len(fields) + 1, 0, 1, 3)
        return group, value_labels

    @staticmethod
    def update_quality_score_labels(value_labels, result_a, result_b):
        if "gradient" in value_labels:
            return update_image_quality_labels(value_labels, result_a, result_b)
        metrics_a = None if result_a is None else result_a["metrics"]
        metrics_b = None if result_b is None else result_b["metrics"]
        for metric_name, labels in value_labels.items():
            for label in labels:
                label.setStyleSheet("color:#c9d1d9; font-weight:400;")
                label.setToolTip("")
            if metrics_a is None or metrics_b is None:
                for label in labels:
                    label.setText("--")
                continue
            value_a = metrics_a.get(metric_name)
            value_b = metrics_b.get(metric_name)
            labels[0].setText(
                "Collecting..." if value_a is None else f"{float(value_a):.2f}"
            )
            labels[1].setText(
                "Collecting..." if value_b is None else f"{float(value_b):.2f}"
            )
            if value_a is not None and value_b is not None:
                # Compare displayed precision so visually equal scores remain neutral.
                score_a = float(f"{float(value_a):.2f}")
                score_b = float(f"{float(value_b):.2f}")
                if np.isfinite(score_a) and np.isfinite(score_b) and score_a != score_b:
                    winner = labels[0 if score_a > score_b else 1]
                    winner.setStyleSheet("color:#55b5ff; font-weight:700;")
                    winner.setToolTip("Higher score for this metric")

    def create_single_quality_score_group(
        self,
        title,
        include_temporal=False,
        collapsible=True,
    ):
        if not include_temporal:
            return create_image_quality_group(self, title, collapsible=collapsible)
        group, layout = self.create_quality_score_container(
            title,
            include_temporal,
            collapsible,
        )
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(7)
        fields = [
            ("Total Score", "overall_score"),
            ("Detail Clarity", "detail_clarity_score"),
            ("Contrast Quality", "contrast_quality_score"),
            ("Flat Cleanliness", "flat_cleanliness_score"),
            ("Thermal Uniformity", "thermal_uniformity_score"),
            ("Clipping Control", "clipping_control_score"),
            ("Artifact Control", "artifact_control_score"),
        ]
        if include_temporal:
            fields.extend(
                [
                    ("Flicker Control", "flicker_control_score"),
                    ("Temporal Stability", "temporal_stability_score"),
                ]
            )
        else:
            fields.append(("Quality Confidence", "quality_confidence_score"))
        value_labels = {}
        for row, (display_name, metric_name) in enumerate(fields):
            metric_label = QLabel(display_name)
            layout.addWidget(metric_label, row, 0)
            value_label = QLabel("--")
            value_label.setProperty("viewerStatValue", True)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setMinimumWidth(90)
            layout.addWidget(value_label, row, 1)
            value_labels[metric_name] = value_label
        if include_temporal:
            note_text = (
                "UVC Lightweight Quality：以低負載空間評分加上 Flicker / Temporal Stability。"
                "Total Score 權重由 Setting 設定。"
            )
        else:
            note_text = (
                "Image Quality：使用原始位深完整分析，Flat 區依灰階與連通區域分開統計；"
                "分數適合相同場景相對比較，不作為絕對 PASS/FAIL 規格。"
            )
        note = self.create_quality_help_note(note_text, include_temporal)
        layout.addWidget(note, len(fields), 0, 1, 2)
        return group, value_labels

    @staticmethod
    def update_single_quality_score_labels(value_labels, result):
        if "gradient" in value_labels:
            return update_image_quality_labels(value_labels, result)
        metrics = None if result is None else result["metrics"]
        for metric_name, label in value_labels.items():
            if metrics is None:
                label.setText("--")
                continue
            value = metrics.get(metric_name)
            label.setText(
                "Collecting..." if value is None else f"{float(value):.2f}"
            )

    def set_form_parameter_visible(self, form, widget, visible):
        widget.setVisible(visible)
        label = form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def rebuild_pipeline_preview_combo(self, editor, combo, include_original=False):
        had_items = combo.count() > 0
        previous_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if include_original:
            combo.addItem("Original / Input Preview", "original")
            combo.addItem("Tone Mapping Output", 0)
        combo.addItem("Final Pipeline Output", None)
        for pipeline_index, stage in enumerate(editor.stages(), start=1):
            if stage.enabled:
                combo.addItem(
                    f"After Step {pipeline_index}: "
                    f"{algorithm_display_name(stage.algorithm_key)}",
                    pipeline_index,
                )
        target_index = combo.findData(previous_data) if had_items else -1
        combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        combo.blockSignals(False)
