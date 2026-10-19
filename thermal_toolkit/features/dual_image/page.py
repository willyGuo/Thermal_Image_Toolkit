"""features / dual_image / page."""

from datetime import datetime
from pathlib import Path
import json
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.analysis.quality.api import quality_analysis_manifest
from thermal_toolkit.analysis.quality.fieldscale_iqa import image_quality_analysis, quality_text
from thermal_toolkit.core.constants import DUAL_UVC_IMAGE_NAME_OPTIONS
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import (
    QApplication,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    Qt,
)
from thermal_toolkit.io.export import save_csv_matrix, save_png, save_rgb_png
from thermal_toolkit.io.raw import IMAGE_INPUT_FILTER, IMAGE_INPUT_NOTE
from thermal_toolkit.reports.quality import save_quality_excel, write_quality_summary
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox, PathRow


class DualImagePageMixin:
    """features / dual_image / page behavior composed by the main window."""

    def build_dual_image_page(self):
        # Dual Image Compare
        p, f = self._form_page()
        self.raw_quality_file_a = PathRow(
            "file",
            IMAGE_INPUT_FILTER,
            "Select RAW / Image Image A for Quality Compare",
        )
        self.raw_quality_file_b = PathRow(
            "file",
            IMAGE_INPUT_FILTER,
            "Select RAW / Image Image B for Quality Compare",
        )
        image_input_note = QLabel(IMAGE_INPUT_NOTE)
        image_input_note.setWordWrap(True)
        image_input_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        f.addRow(image_input_note)
        self.raw_quality_note = QLabel(
            "Use two images of the same scene, resolution, and Bit Depth. Image Quality "
            "uses native DN and separates flat regions by gray level + connected area before "
            "calculating typical/worst noise and uniformity, so different flat objects are not mixed."
        )
        self.raw_quality_note.setWordWrap(True)
        self.raw_quality_note.setStyleSheet(
            "color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; "
            "border-radius:5px; padding:8px;"
        )
        f.addRow(self.raw_quality_note)
        self.dual_image_name_a = FocusWheelComboBox()
        self.dual_image_name_b = FocusWheelComboBox()
        for name, tag in DUAL_UVC_IMAGE_NAME_OPTIONS:
            self.dual_image_name_a.addItem(name, tag)
            self.dual_image_name_b.addItem(name, tag)
        self.dual_image_custom_name_a = QLineEdit()
        self.dual_image_custom_name_b = QLineEdit()
        name_fields = []
        for combo, custom, fallback in (
            (self.dual_image_name_a, self.dual_image_custom_name_a, "Image A"),
            (self.dual_image_name_b, self.dual_image_custom_name_b, "Image B"),
        ):
            combo.addItem("Custom...", "custom")
            custom.setPlaceholderText(f"Enter a name (default: {fallback})")
            custom.setMaxLength(80)
            custom.hide()
            field = QWidget()
            field_layout = QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(4)
            field_layout.addWidget(combo)
            field_layout.addWidget(custom)
            name_fields.append(field)
        self.dual_image_name_b.setCurrentIndex(1)
        f.addRow("Image A Name", name_fields[0])
        f.addRow("RAW / Image A", self.raw_quality_file_a)
        f.addRow("Image B Name", name_fields[1])
        f.addRow("RAW / Image B", self.raw_quality_file_b)
        f.addRow(self.dual_compare_controls)
        (
            self.raw_quality_score_group,
            self.raw_quality_score_labels,
        ) = self.create_quality_score_group("Image Quality Metrics")
        f.addRow(self.raw_quality_score_group)
        self.dual_image_name_a.currentIndexChanged.connect(self.dual_image_name_changed)
        self.dual_image_name_b.currentIndexChanged.connect(self.dual_image_name_changed)
        self.dual_image_custom_name_a.textChanged.connect(self.dual_image_name_changed)
        self.dual_image_custom_name_b.textChanged.connect(self.dual_image_name_changed)
        self.dual_image_name_changed()
        self.stack.addWidget(p)
        self.raw_quality_file_a.path_selected.connect(
            self.auto_preview_raw_quality
        )
        self.raw_quality_file_b.path_selected.connect(
            self.auto_preview_raw_quality
        )
        for path_row in (self.raw_quality_file_a, self.raw_quality_file_b):
            path_row.edit.textChanged.connect(lambda: self.set_image_save_ready("dual", False))
        self.raw_quality_file_a.edit.returnPressed.connect(
            self.auto_preview_raw_quality
        )
        self.raw_quality_file_b.edit.returnPressed.connect(
            self.auto_preview_raw_quality
        )
        self.resolution_combo.currentIndexChanged.connect(
            self.refresh_raw_quality_settings
        )
        self.bit_depth_combo.currentIndexChanged.connect(
            self.refresh_raw_quality_settings
        )

    def dual_image_names(self):
        return tuple(
            (custom.text().strip() or fallback)
            if combo.currentData() == "custom" else combo.currentText()
            for combo, custom, fallback in (
                (self.dual_image_name_a, self.dual_image_custom_name_a, "Image A"),
                (self.dual_image_name_b, self.dual_image_custom_name_b, "Image B"),
            )
        )

    def dual_image_name_changed(self, _index=None):
        for combo, custom in (
            (self.dual_image_name_a, self.dual_image_custom_name_a),
            (self.dual_image_name_b, self.dual_image_custom_name_b),
        ):
            custom.setVisible(combo.currentData() == "custom")
        names = self.dual_image_names()
        for label, name in zip(self.raw_quality_score_group.comparison_header_labels, names):
            label.setText(name)
        if self.FUNCTIONS[self.current_function_index()] == "Dual Image Compare":
            for preview, caption, name in zip(
                (self.preview_original, self.preview_result),
                (self.preview_original_caption, self.preview_result_caption), names
            ):
                preview.set_title(name)
                caption.setText(name)
            self.update_image_preview_titles()
            self.update_image_preview_window(self.preview_original)

    def auto_preview_raw_quality(self, _path=None):
        if (
            self.current_function_index()
            != self.FUNCTIONS.index("Dual Image Compare")
            or not self.raw_quality_file_a.text().strip()
            or not self.raw_quality_file_b.text().strip()
        ):
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.run_raw_quality_compare(save_output=False)
        except Exception as exc:
            self.log.append(f"\nERROR: {exc}")
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def refresh_raw_quality_settings(self, _value=None):
        if (
            self.current_function_index()
            == self.FUNCTIONS.index("Dual Image Compare")
            and self.raw_quality_file_a.text().strip()
            and self.raw_quality_file_b.text().strip()
        ):
            self.auto_preview_raw_quality()

    def run_raw_quality_compare(self, save_output=True):
        self.set_image_save_ready("dual", False)
        name_a, name_b = self.dual_image_names()
        path_a = self.raw_quality_file_a.text().strip()
        path_b = self.raw_quality_file_b.text().strip()
        raw_a, info_a = self.load_analysis_source(path_a)
        raw_b, info_b = self.load_analysis_source(path_b)
        self.dual_image_source_infos = (info_a, info_b)
        self.update_image_preview_titles()
        display_a = raw_to_display_u8(raw_a, info_a["value_max"])
        display_b = raw_to_display_u8(raw_b, info_b["value_max"])
        result_a = image_quality_analysis(raw_a, info_a["value_max"])
        result_b = image_quality_analysis(raw_b, info_b["value_max"])
        self.raw_quality_result_a = result_a
        self.raw_quality_result_b = result_b
        self.set_image_save_ready("dual", True)
        self.current_original = display_a
        self.current_result = display_b
        self.preview_original.set_array(display_a)
        self.preview_result.set_array(display_b)
        self.update_quality_score_labels(
            self.raw_quality_score_labels, result_a, result_b
        )
        self.log.append(f"{name_a}: " + quality_text(result_a))
        self.log.append(f"{name_b}: " + quality_text(result_b))
        if not save_output:
            return

        out = self.make_output("raw_quality_compare")
        outputs = {
            "image_a": "raw_a.png",
            "image_b": "raw_b.png",
            "comparison": "image_comparison.png",
            "display_a": "display_a.png",
            "display_b": "display_b.png",
            "gradient_a": "gradient_a.png",
            "gradient_b": "gradient_b.png",
            "excel": "quality_analysis.xlsx",
        }
        save_png(out / outputs["image_a"], display_a)
        save_png(out / outputs["image_b"], display_b)
        if info_a["source_type"] == "raw":
            outputs["raw_csv_a"] = "raw_a_dn.csv"
            save_csv_matrix(out / outputs["raw_csv_a"], raw_a, "%d")
            self.log.append(f"Saved RAW A DN CSV: {out / outputs['raw_csv_a']}")
        if info_b["source_type"] == "raw":
            outputs["raw_csv_b"] = "raw_b_dn.csv"
            save_csv_matrix(out / outputs["raw_csv_b"], raw_b, "%d")
            self.log.append(f"Saved RAW B DN CSV: {out / outputs['raw_csv_b']}")
        if self.dual_compare_mode.currentIndex() == 1:
            self.save_dual_swipe_preview(out / outputs["comparison"])
        else:
            comparison = self.compose_dual_image_preview([display_a, display_b])
            save_rgb_png(out / outputs["comparison"], cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
        self.log.append(f"Saved comparison preview: {out / outputs['comparison']}")
        save_png(out / outputs["display_a"], result_a["display_u8"])
        save_png(out / outputs["display_b"], result_b["display_u8"])
        save_png(out / outputs["gradient_a"], result_a["gradient_u8"])
        save_png(out / outputs["gradient_b"], result_b["gradient_u8"])
        manifest = {
            "format_version": 1,
            "mode": "raw_quality_compare",
            "comparison_view": self.dual_compare_mode.currentText(),
            "swipe_position": self.dual_swipe_preview.position,
            "analyzed_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "method": "Fieldscale IQA: independent image Gradient/Entropy; paired metrics unavailable without a reference",
            "source_a": str(Path(path_a).resolve()),
            "source_b": str(Path(path_b).resolve()),
            "image_name_a": name_a,
            "image_name_b": name_b,
            "source_info_a": info_a,
            "source_info_b": info_b,
            "quality_a": quality_analysis_manifest(result_a),
            "quality_b": quality_analysis_manifest(result_b),
            "outputs": outputs,
        }
        with open(out / "quality.json", "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
        with open(
            out / "quality_summary.txt", "w", encoding="utf-8"
        ) as summary:
            summary.write(f"Analyzed: {manifest['analyzed_at']}\n")
            summary.write(f"RAW A: {manifest['source_a']}\n")
            summary.write(f"RAW B: {manifest['source_b']}\n")
            summary.write(
                f"Input A: {info_a}\nInput B: {info_b}\n"
            )
            summary.write(
                "Use these scores only for same-scene relative comparison.\n"
            )
            write_quality_summary(summary, name_a, result_a)
            write_quality_summary(summary, name_b, result_b)
        save_quality_excel(
            out / outputs["excel"],
            "Dual Image Compare",
            {
                key: value
                for key, value in manifest.items()
                if key not in {"quality_a", "quality_b"}
            },
            [
                {
                    "label": name_a,
                    "metrics": result_a["metrics"],
                    "unavailable_reasons": result_a.get("unavailable_reasons", {}),
                },
                {
                    "label": name_b,
                    "metrics": result_b["metrics"],
                    "unavailable_reasons": result_b.get("unavailable_reasons", {}),
                },
            ],
        )
        self.log.append(f"Saved image quality comparison: {out}")
