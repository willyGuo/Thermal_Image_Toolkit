"""features / bpr / page."""

import csv
import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.algorithms.parameters import BPR_HIGHLIGHT_RADIUS, BPR_MARK_RADIUS
from thermal_toolkit.core.qt import QFont, QLabel, QTextEdit
from thermal_toolkit.features.bpr.coordinates import (
    format_bpr_coordinate_report,
    load_bpr_coordinates,
    split_coordinates_by_bounds,
)
from thermal_toolkit.features.bpr.render import draw_bpr_markers
from thermal_toolkit.io.export import save_rgb_png
from thermal_toolkit.io.raw import IMAGE_INPUT_FILTER
from thermal_toolkit.ui.widgets.groups import QualityScoreCheckBox
from thermal_toolkit.ui.widgets.inputs import FocusWheelSpinBox, PathRow


class BprPageMixin:
    """features / bpr / page behavior composed by the main window."""

    def build_bpr_page(self):
        # BPR Bad Pixel Marking
        p, f = self._form_page()
        self.bpr_form = f
        self.bpr_coordinate_file = PathRow(
            "file",
            "BPR Table / Coordinates (*.bin *.BIN *.csv *.CSV *.txt *.TXT)",
            "Select Novatek dpc_table.bin or x,y coordinate file",
        )
        self.bpr_use_raw = QualityScoreCheckBox("Overlay circles on an image")
        background_font = self.bpr_use_raw.font()
        background_font.setWeight(QFont.Normal)
        self.bpr_use_raw.setFont(background_font)
        self.bpr_use_raw.setMinimumWidth(
            self.bpr_use_raw.fontMetrics().horizontalAdvance(self.bpr_use_raw.text()) + 44
        )
        self.bpr_use_raw.setChecked(False)
        self.bpr_raw_file = PathRow(
            "file",
            IMAGE_INPUT_FILTER,
            "Select optional image background",
        )
        self.bpr_circle_radius = FocusWheelSpinBox()
        self.bpr_circle_radius.setRange(1, 20)
        self.bpr_circle_radius.setValue(BPR_MARK_RADIUS)
        self.bpr_highlight_bright_points = QualityScoreCheckBox(
            "Highlight bright points"
        )
        highlight_font = self.bpr_highlight_bright_points.font()
        highlight_font.setWeight(QFont.Normal)
        self.bpr_highlight_bright_points.setFont(highlight_font)
        self.bpr_highlight_radius = FocusWheelSpinBox()
        self.bpr_highlight_radius.setRange(1, 20)
        self.bpr_highlight_radius.setValue(BPR_HIGHLIGHT_RADIUS)
        self.bpr_note = QLabel()
        self.bpr_note.setWordWrap(True)
        self.bpr_note.setStyleSheet(
            "color:#c8d0d9; background:#292d33; border-left:3px solid #79afd3; "
            "border-radius:5px; padding:8px;"
        )
        self.bpr_coordinate_info = QTextEdit()
        self.bpr_coordinate_info.setReadOnly(True)
        self.bpr_coordinate_info.setMinimumHeight(190)
        self.bpr_coordinate_info.setPlaceholderText(
            "Browse a coordinate file to preview bad-pixel locations."
        )
        self.update_log_font_size(self.app_settings["log_font_size"])
        for row in (self.bpr_coordinate_file, self.bpr_raw_file):
            row.edit.textChanged.connect(self.invalidate_bpr_preview)
            row.path_selected.connect(self.auto_preview_bpr)
            row.edit.returnPressed.connect(self.auto_preview_bpr)
        self.bpr_use_raw.toggled.connect(self.auto_preview_bpr)
        self.bpr_circle_radius.valueChanged.connect(self.auto_preview_bpr)
        self.bpr_highlight_bright_points.toggled.connect(
            self.auto_preview_bpr
        )
        self.bpr_highlight_bright_points.toggled.connect(
            self.update_bpr_marking_ui
        )
        self.bpr_highlight_radius.valueChanged.connect(self.auto_preview_bpr)
        self.bit_depth_combo.currentIndexChanged.connect(self.auto_preview_bpr)
        self.bpr_use_raw.toggled.connect(self.update_bpr_marking_ui)
        self.resolution_combo.currentIndexChanged.connect(self.auto_preview_bpr)
        self.resolution_combo.currentIndexChanged.connect(self.update_bpr_coordinate_note)
        f.addRow("BPR Table / Coordinates", self.bpr_coordinate_file)
        f.addRow("Background", self.bpr_use_raw)
        f.addRow("Image File", self.bpr_raw_file)
        f.addRow("Red Circle Radius", self.bpr_circle_radius)
        f.addRow("Bright Point", self.bpr_highlight_bright_points)
        f.addRow("Bright Point Radius", self.bpr_highlight_radius)
        f.addRow(self.bpr_note)
        f.addRow("Bad Pixel Information", self.bpr_coordinate_info)
        self.stack.addWidget(p)
        self.update_bpr_marking_ui()
        self.update_bpr_coordinate_note()

    def update_bpr_marking_ui(self, _checked=None):
        self.set_form_parameter_visible(
            self.bpr_form,
            self.bpr_raw_file,
            self.bpr_use_raw.isChecked(),
        )
        self.set_form_parameter_visible(
            self.bpr_form,
            self.bpr_highlight_radius,
            self.bpr_highlight_bright_points.isChecked(),
        )

    def update_bpr_coordinate_note(self, _index=None):
        self.bpr_note.setText(
            "座標方向：左上角為原點 (0,0)；X 由左往右增加，"
            f"有效範圍 0..{self.width - 1}；Y 由上往下增加，"
            f"有效範圍 0..{self.height - 1}。座標以 (x,y) 表示，"
            "對應程式陣列 RAW[y,x]。例如 (600,79) 是從左側第 601 欄、"
            "從上方第 80 列。Novatek BIN 使用 SIE row 格式；CSV/TXT 使用 x,y；"
            "重複座標只圈選一次。"
        )

    def clear_bpr_coordinate_info(self, _value=None):
        self.bpr_coordinate_info.clear()

    def invalidate_bpr_preview(self, _value=None):
        self._bpr_preview = None
        self.bpr_coordinate_info.clear()
        if self.FUNCTIONS[self.current_function_index()] == "BPR Bad Pixel Marking":
            self.run_btn.setEnabled(False)
            self.preview_original.set_array(None)
            self.preview_result.set_array(None)
            self.current_original = self.current_result = None

    def auto_preview_bpr(self, _value=None):
        self.invalidate_bpr_preview()
        if self.FUNCTIONS[self.current_function_index()] != "BPR Bad Pixel Marking":
            return
        if not self.bpr_coordinate_file.text().strip():
            return
        if self.bpr_use_raw.isChecked() and not self.bpr_raw_file.text().strip():
            return
        try:
            coordinate_path = self.bpr_coordinate_file.text()
            coordinates, table_info = load_bpr_coordinates(coordinate_path)
            raw_path = ""
            if self.bpr_use_raw.isChecked():
                raw_path = self.bpr_raw_file.text()
                raw, info = self.load_analysis_source(raw_path)
                base_gray = raw_to_display_u8(raw, info["value_max"])
                background_mode = f"{info['value_name']} fixed scale 0..{info['value_max']}"
                base_filename = "image_fixed.png"
            else:
                base_gray = np.full((self.height, self.width), 255, dtype=np.uint8)
                background_mode = "White canvas"
                base_filename = "white_canvas.png"
            height, width = base_gray.shape
            valid_coordinates, outside_coordinates = split_coordinates_by_bounds(coordinates, width, height)
            coordinate_report = format_bpr_coordinate_report(table_info, coordinates, outside_coordinates, width, height)
            self.bpr_coordinate_info.setPlainText(coordinate_report)
            if not valid_coordinates:
                raise ValueError(f"No BPR coordinates are inside the {width}x{height} image.")
            radius = self.bpr_circle_radius.value()
            highlight_bright_points = (
                self.bpr_highlight_bright_points.isChecked()
            )
            highlight_radius = self.bpr_highlight_radius.value()
            base_rgb = np.repeat(base_gray[:, :, np.newaxis], 3, axis=2)
            marked_rgb = draw_bpr_markers(
                base_gray,
                valid_coordinates,
                radius,
                highlight_bright_points,
                highlight_radius,
            )
            self._bpr_preview = (coordinate_path, coordinates, table_info, outside_coordinates,
                                 coordinate_report, raw_path, background_mode, base_filename,
                                 width, height, radius, highlight_bright_points,
                                 highlight_radius, base_rgb, marked_rgb)
            self.current_original, self.current_result = base_rgb, marked_rgb
            self.preview_original.set_array(base_rgb)
            self.preview_result.set_array(marked_rgb)
            self.run_btn.setEnabled(True)
            self.bpr_note.setText(
                f"Coordinates: top-left (0,0), X=0..{width-1}, Y=0..{height-1}; image[y,x]. "
                "Preview updates automatically. Save writes the displayed result."
            )
            self.log.append(f"BPR preview: {width}x{height}, marked={len(valid_coordinates)}, outside={len(outside_coordinates)}; {background_mode}")
        except Exception as exc:
            self.log.append(f"BPR preview error: {exc}")

    def run_bpr_marking(self):
        cached = getattr(self, "_bpr_preview", None)
        if cached is None:
            return
        (coordinate_path, coordinates, table_info, outside_coordinates,
         coordinate_report, raw_path, background_mode, base_filename,
         width, height, radius, highlight_bright_points, highlight_radius,
         base_rgb, marked_rgb) = cached
        out = self.make_output("bpr_bad_pixel_marking")
        base_path = out / base_filename
        marked_path = out / "bad_pixels_marked.png"
        coordinate_output_path = out / "bad_pixel_coordinates.csv"
        summary_path = out / "bpr_marking_summary.txt"

        save_rgb_png(base_path, base_rgb)
        save_rgb_png(marked_path, marked_rgb)
        outside_set = set(outside_coordinates)
        with open(coordinate_output_path, "w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            writer.writerow(["x", "y", "in_bounds", "index"])
            for index, (x, y) in enumerate(coordinates, 1):
                writer.writerow([x, y, 0 if (x, y) in outside_set else 1, index])

        with open(summary_path, "w", encoding="utf-8") as summary:
            summary.write(f"BPR source: {coordinate_path}\n")
            summary.write(f"Source format: {table_info['format']}\n")
            summary.write(f"Resolution: {width}x{height}\n")
            summary.write(f"Background: {background_mode}\n")
            if raw_path:
                summary.write(f"Image source: {raw_path}\n")
            summary.write(f"Circle radius: {radius} px\n")
            summary.write(
                "Bright point highlight: "
                f"{'enabled' if highlight_bright_points else 'disabled'}\n"
            )
            if highlight_bright_points:
                summary.write(f"Bright point radius: {highlight_radius} px\n")
            summary.write("\n")
            summary.write(coordinate_report)
            summary.write("\n")

        self.log.append(f"Saved: {marked_path}")
        self.log.append(f"Coordinates: {coordinate_output_path}")
