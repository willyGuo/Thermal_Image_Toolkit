"""Static-image metric rows shared by the single and comparison views."""

from thermal_toolkit.analysis.quality.fieldscale_iqa import FIELDS, HELP
from thermal_toolkit.core.qt import QLabel, Qt


def create_image_quality_group(owner, title, paired=False, collapsible=True):
    group, layout = owner.create_quality_score_container(title, False, collapsible)
    labels = {}
    start = 1 if paired else 0
    if paired:
        headers = []
        for col, name in enumerate(("Metric", "A", "B")):
            label = QLabel(name)
            label.setStyleSheet("font-weight:700; color:#c9e2f2;")
            layout.addWidget(label, 0, col)
            if col:
                headers.append(label)
        group.comparison_header_labels = headers
    for row, (name, key) in enumerate(FIELDS, start=start):
        title_label = QLabel(name)
        title_label.setToolTip(HELP[key])
        layout.addWidget(title_label, row, 0)
        values = []
        for col in range(1, 3 if paired else 2):
            label = QLabel("--")
            label.setProperty("viewerStatValue", True)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setToolTip(HELP[key])
            layout.addWidget(label, row, col)
            values.append(label)
        labels[key] = values if paired else values[0]
    note = owner.create_quality_help_note(
        "Gradient / Entropy 越高越好；C_global / C_local 越低越好；TMQI 越高越好。"
        "後三項需要對應原圖，無參考時顯示 N/A；不合併為總分。")
    layout.addWidget(note, len(FIELDS) + start, 0, 1, 3 if paired else 2)
    return group, labels


def update_image_quality_labels(labels, *results):
    for key, widgets in labels.items():
        widgets = widgets if isinstance(widgets, list) else [widgets]
        displayed = []
        for label, result in zip(widgets, results):
            value = None if result is None else result["metrics"].get(key)
            label.setText("--" if result is None else "N/A" if value is None else f"{value:.5f}")
            reason = "" if result is None else result.get("unavailable_reasons", {}).get(key, "")
            label.setToolTip(HELP[key] + ("\n" + reason if reason else ""))
            label.setStyleSheet("color:#c9d1d9; font-weight:400;")
            displayed.append(None if value is None else round(value, 5))
        if len(displayed) == 2 and all(value is not None for value in displayed):
            a, b = displayed
            if a != b:
                winner = int(b < a if key in ("c_global", "c_local") else b > a)
                widgets[winner].setStyleSheet("color:#55b5ff; font-weight:700;")
