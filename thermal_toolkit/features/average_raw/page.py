"""features / average_raw / page."""

from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.core.constants import RAW_DTYPE
from thermal_toolkit.io.export import save_png
from thermal_toolkit.io.raw import average_raw_folder, load_raw
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox, PathRow


class AverageRawPageMixin:
    """features / average_raw / page behavior composed by the main window."""

    def build_average_raw_page(self):
        # Average
        p, f = self._form_page(); self.avg_folder = PathRow("folder", dialog_title="Select RAW Folder to Average"); self.avg_suffix_combo = FocusWheelComboBox(); self.avg_suffix_combo.addItems(["LOW", "HIGH"]); f.addRow("RAW Folder", self.avg_folder); f.addRow("Filename Suffix", self.avg_suffix_combo); self.stack.addWidget(p)

    def run_average(self):
        avg, valid, skipped = average_raw_folder(self.avg_folder.text(), self.width, self.height)
        out = self.make_output("average_raw")
        suffix = self.avg_suffix_combo.currentText()
        avg_preview = raw_to_display_u8(avg, self.raw_max)
        raw_path = out / f"average_{suffix}.raw"; avg.astype(RAW_DTYPE).tofile(raw_path)
        save_png(out / f"average_preview_{suffix}.png", avg_preview)
        with open(out / "source_files.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(valid))
        self.current_original = load_raw(valid[0], self.width, self.height); self.current_result = avg
        self.preview_result.set_array(avg_preview)
        self.log.append(f"Valid frames: {len(valid)}; Skipped: {len(skipped)}")
        self.log.append(f"Filename suffix: {suffix}")
        self.log_stats("First Frame", self.current_original); self.log_stats("Average RAW", avg)
        self.log.append(f"Preview scale: fixed 0..{self.raw_max} DN")
        self.log.append(f"Saved: {raw_path}")
