import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from thermal_toolkit.core.qt import QApplication, QFormLayout, QStackedWidget, QWidget
from thermal_toolkit.features.nuc.calibration import build_novatek_nuc_tables, apply_novatek_nuc
from thermal_toolkit.features.nuc.calculation_page import NucCalculationPageMixin


class NucTablesTests(unittest.TestCase):
    def test_q8_and_q14_known_gain_and_b(self):
        low = np.array([[100, 220]], dtype=np.uint16)
        high = np.array([[300, 620]], dtype=np.uint16)
        for n in (8, 14):
            k, b, valid, low_mean, high_mean = build_novatek_nuc_tables(low, high, 50, n)
            np.testing.assert_array_equal(k, [[3 * (1 << n) // 2, 3 * (1 << n) // 4]])
            np.testing.assert_array_equal(b, [[-7, 7]])
            self.assertTrue(valid.all())
            self.assertEqual((low_mean, high_mean), (160, 460))
            np.testing.assert_array_equal(apply_novatek_nuc(high, k, b, 16383, n), [[460, 459]])
        for actual, expected in zip(build_novatek_nuc_tables(low, high, 50),
                                    build_novatek_nuc_tables(low, high, 50, 8)):
            np.testing.assert_array_equal(actual, expected)

    def test_integer_reference_across_all_supported_n(self):
        low = np.array([[100, 250, 800], [600, 1000, 1500]], dtype=np.uint16)
        high = np.array([[101, 700, 1300], [900, 7000, 4000]], dtype=np.uint16)
        for n in range(16):
            k, b, valid, lo, hi = build_novatek_nuc_tables(low, high, 50, n)
            for index in np.ndindex(low.shape):
                delta = int(high[index]) - int(low[index])
                raw_k = ((hi - lo) * (1 << n)) // delta if delta > 50 else 0
                expected_valid = raw_k > 0
                expected_k = min(raw_k, 65535) if expected_valid else 1 << n
                old_b = lo - expected_k * int(low[index]) // (1 << n)
                numerator = -old_b * (1 << n)
                expected_b = (abs(numerator) + expected_k // 2) // expected_k
                expected_b *= -1 if numerator < 0 else 1
                expected_b = max(-32768, min(32767, expected_b)) if expected_valid else 0
                self.assertEqual(int(k[index]), expected_k)
                self.assertEqual(int(b[index]), expected_b)
                self.assertEqual(bool(valid[index]), expected_valid)

    def test_saturation_and_invalid_identity(self):
        low = np.array([[100, 100, 100]], dtype=np.uint16)
        high = np.array([[100, 102, 10000]], dtype=np.uint16)
        for n in (0, 8, 14, 15):
            k, b, valid, _, _ = build_novatek_nuc_tables(low, high, 1, n)
            self.assertFalse(valid[0, 0])
            self.assertEqual(int(k[0, 0]), 1 << n)
            self.assertEqual(int(b[0, 0]), 0)
            self.assertEqual(int(apply_novatek_nuc(low, k, b, 16383, n)[0, 0]), 100)
            if n >= 8:
                self.assertEqual(int(k[0, 1]), 65535)
        for n in (-1, 16, 31, 8.5):
            with self.assertRaises(ValueError):
                build_novatek_nuc_tables(low, high, 1, n)


class CalculationHarness(QWidget, NucCalculationPageMixin):
    width = height = 2
    raw_max = 16383

    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget(self)
        self.preview_original = Mock()
        self.preview_result = Mock()
        self.log = Mock()
        self.build_nuc_calculation_page()

    def _form_page(self):
        page = QWidget()
        return page, QFormLayout(page)


class NucExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selected_n_drives_export_and_preview(self):
        widget = CalculationHarness()
        self.addCleanup(widget.close)
        self.assertEqual(widget.nuc_calculate_gain_shift.value(), 8)
        self.assertFalse(widget.nuc_calculate_gain_shift.keyboardTracking())
        low = np.array([[100, 220], [100, 220]], dtype=np.uint16)
        high = np.array([[300, 620], [300, 620]], dtype=np.uint16)
        widget._get_nuc_inputs = lambda: (low, high, ["low"], ["high"], [], [])
        for n in (8, 14):
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                widget.make_output = lambda _: out
                widget.nuc_calculate_gain_shift.setValue(n)
                widget.run_nuc_calculate()
                k = np.fromfile(out / "point1_K.bin", dtype="<u2").reshape(low.shape)
                b = np.fromfile(out / "point1_B.bin", dtype="<i2").reshape(low.shape)
                np.testing.assert_array_equal(k, [[3 * (1 << n) // 2, 3 * (1 << n) // 4]] * 2)
                corrected = apply_novatek_nuc(high, k, b, 16383, n)
                np.testing.assert_array_equal(widget.current_result, corrected)
                np.testing.assert_array_equal(np.fromfile(out / "high_nuc_corrected.raw", dtype="<u2").reshape(low.shape), corrected)
                self.assertIn(f"Q{n}", (out / "nuc_summary.txt").read_text(encoding="utf-8"))
                self.assertIn(f"Q{n}", widget.nuc_calculate_format.text())


if __name__ == "__main__":
    unittest.main()
