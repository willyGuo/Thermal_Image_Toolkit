"""features / nuc / calculation_page."""

import csv
import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.core.constants import NOVATEK_NUC_GAIN_SHIFT, RAW_DTYPE
from thermal_toolkit.core.qt import QLabel
from thermal_toolkit.features.nuc.calibration import apply_novatek_nuc, build_novatek_nuc_tables, calculate_nuc
from thermal_toolkit.io.export import save_csv_matrix, save_png
from thermal_toolkit.io.raw import average_raw_folder, load_raw
from thermal_toolkit.ui.widgets.inputs import FocusWheelComboBox, FocusWheelSpinBox, PathRow


class NucCalculationPageMixin:
    """features / nuc / calculation_page behavior composed by the main window."""

    def build_nuc_calculation_page(self):
        # NUC Calculate
        p, f = self._form_page()
        self.nuc_mode = FocusWheelComboBox(); self.nuc_mode.addItems(["Folder Average (Recommended)", "Single RAW"]); self.nuc_mode.currentIndexChanged.connect(self.update_nuc_mode)
        self.nuc_point_combo = FocusWheelComboBox()
        for point_number in range(1, 6):
            self.nuc_point_combo.addItem(f"Point {point_number}", point_number)
        self.low_single = PathRow("file", "RAW Files (*.raw *.RAW *.bin *.BIN)", "Select LOW RAW File for NUC")
        self.high_single = PathRow("file", "RAW Files (*.raw *.RAW *.bin *.BIN)", "Select HIGH RAW File for NUC")
        self.low_folder = PathRow("folder", dialog_title="Select LOW RAW Folder for NUC")
        self.high_folder = PathRow("folder", dialog_title="Select HIGH RAW Folder for NUC")
        self.min_delta = FocusWheelSpinBox(); self.min_delta.setRange(0, 16383); self.min_delta.setValue(50)
        self.nuc_calculate_gain_shift = FocusWheelSpinBox()
        self.nuc_calculate_gain_shift.setRange(0, 15)
        self.nuc_calculate_gain_shift.setValue(NOVATEK_NUC_GAIN_SHIFT)
        self.nuc_calculate_gain_shift.setToolTip(
            "K uses N fractional bits: N=8 is Q8, N=14 is Q14. "
            "Use the same N in NUC_Analyze and the target correction. "
            "N is limited to 15 so uint16 K can represent unity gain for invalid pixels."
        )
        f.addRow("NUC Point", self.nuc_point_combo); f.addRow("NUC Source Mode", self.nuc_mode); f.addRow("LOW RAW", self.low_single); f.addRow("HIGH RAW", self.high_single); f.addRow("LOW Folder", self.low_folder); f.addRow("HIGH Folder", self.high_folder); f.addRow("Min Delta DN", self.min_delta)
        f.addRow("N (K fractional bits)", self.nuc_calculate_gain_shift)
        self.nuc_calculate_format = QLabel()
        self.nuc_calculate_format.setWordWrap(True)
        f.addRow(self.nuc_calculate_format)
        self.nuc_calculate_gain_shift.valueChanged.connect(self.update_nuc_calculate_format)
        self.update_nuc_calculate_format()
        note = QLabel("Recommended: capture multiple frames at each stable blackbody temperature, average each folder pixel-by-pixel, then calculate NUC Gain/Offset and Novatek K/B tables.")
        note.setWordWrap(True); note.setStyleSheet("color:#c98;"); f.addRow(note)
        self.stack.addWidget(p); self.update_nuc_mode()

    def update_nuc_calculate_format(self):
        n = self.nuc_calculate_gain_shift.value()
        self.nuc_calculate_format.setText(
            f"K: uint16 Q{n}, gain = K / {1 << n}; unity K = {1 << n}.\n"
            f"Maximum gain: {65535 / (1 << n):.8f}; larger K is clipped to 65535. "
            "B: signed int16 DN.\n"
            f"Output = clip((K * (Input - B)) >> {n}, 0, Bit Depth Max). "
            f"Set NUC_Analyze N to {n} for these tables."
        )

    def update_nuc_mode(self):
        folder_mode = self.nuc_mode.currentIndex() == 0
        for w in (self.low_folder, self.high_folder): w.setEnabled(folder_mode)
        for w in (self.low_single, self.high_single): w.setEnabled(not folder_mode)

    def _get_nuc_inputs(self):
        if self.nuc_mode.currentIndex() == 0:
            low, low_files, low_skip = average_raw_folder(self.low_folder.text(), self.width, self.height)
            high, high_files, high_skip = average_raw_folder(self.high_folder.text(), self.width, self.height)
            return low, high, low_files, high_files, low_skip, high_skip
        low = load_raw(self.low_single.text(), self.width, self.height)
        high = load_raw(self.high_single.text(), self.width, self.height)
        return low, high, [self.low_single.text()], [self.high_single.text()], [], []

    def run_nuc_calculate(self):
        n = self.nuc_calculate_gain_shift.value()
        low, high, low_files, high_files, low_skip, high_skip = self._get_nuc_inputs()
        gain, offset, valid, delta, low_avg, high_avg = calculate_nuc(low, high, self.min_delta.value())
        stored_k, gfx_b, table_valid, novatek_low_avg, novatek_high_avg = build_novatek_nuc_tables(
            low, high, self.min_delta.value(), n
        )
        valid = table_valid
        saturated_k_count = int(np.count_nonzero(
            ((novatek_high_avg - novatek_low_avg) << n) // delta[valid].astype(np.int64) > 65535
        ))
        point_number = self.nuc_point_combo.currentData()
        out = self.make_output("nuc_calculate")
        low.astype(RAW_DTYPE).tofile(out / "low_input_used.raw"); high.astype(RAW_DTYPE).tofile(out / "high_input_used.raw")
        np.save(out / "gain.npy", gain); np.save(out / "offset.npy", offset)
        save_csv_matrix(out / "gain.csv", gain, "%.8f"); save_csv_matrix(out / "offset.csv", offset, "%.3f")
        k_path = out / f"point{point_number}_K.bin"
        b_path = out / f"point{point_number}_B.bin"
        stored_k.astype("<u2").tofile(k_path)
        gfx_b.astype("<i2").tofile(b_path)
        yy, xx = np.where(~valid)
        with open(out / "invalid_pixels.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["x", "y", "low", "high", "delta"])
            for y, x in zip(yy, xx): wr.writerow([x, y, int(low[y,x]), int(high[y,x]), float(delta[y,x])])
        with open(out / "nuc_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Novatek NUC point: {point_number}\nMode: {self.nuc_mode.currentText()}\nLow frames: {len(low_files)}\nHigh frames: {len(high_files)}\n")
            f.write(f"Low mean: {low_avg:.6f}\nHigh mean: {high_avg:.6f}\nTarget delta: {high_avg-low_avg:.6f}\n")
            f.write(f"Min delta: {self.min_delta.value()}\nValid pixels: {valid.sum()} / {valid.size}\n")
            if valid.any(): f.write(f"Gain mean/std: {gain[valid].mean():.8f} / {gain[valid].std():.8f}\nOffset mean/std: {offset[valid].mean():.4f} / {offset[valid].std():.4f}\n")
            f.write(f"Novatek integer LOW/HIGH mean: {novatek_low_avg} / {novatek_high_avg}\n")
            f.write(f"N: {n}\nK format: uint16 little-endian Q{n}; gain = K / {1 << n}; unity K = {1 << n}\nB format: signed int16 little-endian DN\n")
            f.write(f"Maximum gain: {65535 / (1 << n):.8f}\nK clipped to 65535 pixels: {saturated_k_count}\n")
            f.write(f"Runtime formula: output = clip((K * (input - B)) >> {n}, 0, {self.raw_max})\n")
            f.write(f"NUC_Analyze N: {n}\n")
            f.write(f"K file: {k_path.name} ({k_path.stat().st_size} bytes)\n")
            f.write(f"B file: {b_path.name} ({b_path.stat().st_size} bytes)\n")
        low_preview = raw_to_display_u8(low, self.raw_max)
        high_preview = raw_to_display_u8(high, self.raw_max)
        save_png(out / "low_preview.png", low_preview); save_png(out / "high_preview.png", high_preview)
        # Preview the same quantized K/B formula used by the Novatek GFX path.
        corrected_high = apply_novatek_nuc(high, stored_k, gfx_b, self.raw_max, n)
        corrected_high_preview = raw_to_display_u8(corrected_high, self.raw_max)
        corrected_high.astype(RAW_DTYPE).tofile(out / "high_nuc_corrected.raw")
        save_png(out / "high_nuc_corrected_preview.png", corrected_high_preview)
        self.current_original = high; self.current_result = corrected_high
        self.preview_original.set_array(high_preview)
        self.preview_result.set_array(corrected_high_preview)
        self.log.append(f"Novatek NUC point={point_number}: {k_path.name}, {b_path.name}")
        self.log.append(f"K=uint16 Q{n}, N={n}, unity K={1 << n}; use N={n} in NUC_Analyze")
        self.log.append(f"K clipped to 65535 pixels: {saturated_k_count:,}")
        self.log.append(f"Mode: {self.nuc_mode.currentText()}")
        self.log.append(f"LOW frames={len(low_files)}, HIGH frames={len(high_files)}, invalid/skipped={len(low_skip)+len(high_skip)}")
        self.log.append(f"Low mean={low_avg:.3f}, High mean={high_avg:.3f}, Target delta={high_avg-low_avg:.3f}")
        self.log.append(f"Valid NUC pixels={valid.sum():,}/{valid.size:,} ({valid.mean()*100:.3f}%)")
        if valid.any(): self.log.append(f"Gain mean={gain[valid].mean():.6f}, std={gain[valid].std():.6f}; Offset mean={offset[valid].mean():.3f}, std={offset[valid].std():.3f}")
        if valid.any():
            corrected_valid = corrected_high[valid].astype(np.float64)
            self.log.append(
                f"Corrected HIGH valid pixels: mean={corrected_valid.mean():.3f}, "
                f"std={corrected_valid.std():.3f}, min={corrected_valid.min():.0f}, "
                f"max={corrected_valid.max():.0f} DN"
            )
        if not valid.all():
            self.log.append(f"Invalid pixels kept uncorrected: {(~valid).sum():,}")
        self.log.append("Preview scale: fixed 0..Bit Depth Max (no percentile stretching)")
