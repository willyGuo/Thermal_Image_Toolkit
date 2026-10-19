"""Regression checks for final native-DN LUT gain limiting."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from thermal_toolkit.algorithms.histogram import histogram_weighted_mapping_native_u8
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping, tone_mapping_parameters
from thermal_toolkit.algorithms.weighted_grid import histogram_weighted_grid_u8, weighted_tiles
from thermal_toolkit.reports.histogram_weighted import save_histogram_weighted_intervals


class WeightedGainLimitTests(unittest.TestCase):
    def setUp(self):
        self.example = np.repeat(
            np.arange(7998, 8003), [100, 300, 500, 300, 100]
        ).reshape(10, 130)

    def test_blackbody_example_and_brightness_anchor(self):
        expected = {
            0: [10, 49, 128, 206, 245],
            1: [126, 127, 128, 129, 130],
            2: [124, 126, 128, 130, 132],
            4: [120, 124, 128, 132, 136],
        }
        for gain, levels in expected.items():
            with self.subTest(gain=gain):
                output, _, info, used = apply_tone_mapping(
                    self.example, "histogram_weighted", 16383,
                    {"histogram_weighted_gain_limit": gain},
                )
                self.assertEqual(np.unique(output).tolist(), levels)
                self.assertEqual(used["histogram_weighted_gain_limit"], gain)
                self.assertEqual(info["output_min"], levels[0])
                self.assertEqual(info["output_max"], levels[-1])
                if gain:
                    self.assertEqual(info["gain_anchor_dn"], 8000)
                    self.assertEqual(info["gain_anchor_gray"], 128)

    def test_old_parameters_default_to_off(self):
        parameters = tone_mapping_parameters("histogram_weighted")
        self.assertEqual(parameters.pop("histogram_weighted_gain_limit"), 0)
        output, _, _, used = apply_tone_mapping(
            self.example, "histogram_weighted", 16383, parameters
        )
        self.assertEqual(used["histogram_weighted_gain_limit"], 0)
        self.assertEqual(np.unique(output).tolist(), [10, 49, 128, 206, 245])

    def test_full_lut_bound_and_weights_unchanged(self):
        rng = np.random.default_rng(930)
        for maximum in (255, 16383, 65535):
            inputs = [
                rng.integers(maximum // 2, maximum // 2 + 9, (17, 31)),
                rng.choice([0, 1, maximum // 3, maximum - 1, maximum], (18, 30)),
                rng.integers(0, maximum + 1, (19, 29)),
            ]
            for source in inputs:
                for clip, redistribute, bright in ((0, False, 1), (0.5, True, 1.05), (2, False, 2)):
                    original = {}
                    histogram_weighted_mapping_native_u8(
                        source, maximum, clip, redistribute, bright, interval_data=original
                    )
                    for gain in (1, 2, 4):
                        with self.subTest(maximum=maximum, clip=clip, gain=gain):
                            data, lut_data = {}, {}
                            output, info = histogram_weighted_mapping_native_u8(
                                source, maximum, clip, redistribute, bright,
                                interval_data=data, lut_data=lut_data, gain_limit=gain,
                            )
                            lut = lut_data["lut"]
                            steps = np.diff(lut.astype(np.int64))
                            self.assertTrue(np.all((0 <= steps) & (steps <= gain)))
                            np.testing.assert_array_equal(output, lut[source])
                            np.testing.assert_array_equal(lut, data["final_lut"])
                            for key in ("original_weights", "before_bright_weights", "adjusted_weights", "before_gain_lut"):
                                np.testing.assert_array_equal(data[key], original[key])
                            anchor = np.sort(source.ravel())[(source.size - 1) // 2]
                            self.assertEqual(info["gain_anchor_dn"], anchor)
                            self.assertEqual(lut[anchor], original["final_lut"][anchor])
                            if not (clip > 0 and redistribute):
                                self.assertTrue(np.all(lut[:source.min() + 1] == lut[source.min()]))
                                self.assertTrue(np.all(lut[source.max():] == lut[source.max()]))

    def test_already_small_steps_are_unchanged(self):
        source = np.arange(65536).reshape(256, 256)
        original, _ = histogram_weighted_mapping_native_u8(source, 65535)
        for gain in (1, 2, 4):
            output, _ = histogram_weighted_mapping_native_u8(source, 65535, gain_limit=gain)
            np.testing.assert_array_equal(output, original)

    def test_empty_and_constant_keep_existing_output(self):
        for source in (np.empty((0, 0), dtype=np.uint16), np.full((3, 5), 8000)):
            for gain in (0, 1, 2, 4):
                data, lut_data = {}, {}
                output, info = histogram_weighted_mapping_native_u8(
                    source, 16383, interval_data=data, lut_data=lut_data, gain_limit=gain
                )
                np.testing.assert_array_equal(output, np.zeros(source.shape, dtype=np.uint8))
                self.assertFalse(lut_data["lut"].any())
                self.assertFalse(data["final_lut"].any())
                self.assertIsNone(info["gain_anchor_dn"])

    def test_invalid_gain_is_rejected(self):
        for gain in (-1, 0.05, 4.1, 8, float("nan"), float("inf")):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                histogram_weighted_mapping_native_u8(self.example, 16383, gain_limit=gain)

    def test_fractional_gain_accumulates_before_output_rounding(self):
        data = {}
        output, info = histogram_weighted_mapping_native_u8(
            self.example, 16383, gain_limit=0.5, interval_data=data
        )
        np.testing.assert_array_equal(
            data["limited_lut_float"][7998:8003], [127, 127.5, 128, 128.5, 129]
        )
        np.testing.assert_array_equal(data["final_lut"][7998:8003], [127, 128, 128, 129, 129])
        self.assertEqual(info["gain_limit"], 0.5)
        self.assertEqual(info["gain_anchor_gray"], 128)
        self.assertEqual(int(np.ptp(output)), 2)

    def test_all_tenths_preserve_monotonicity_and_unrounded_bound(self):
        rng = np.random.default_rng(4)
        for maximum in (255, 16383, 65535):
            source = rng.integers(maximum // 2, maximum // 2 + 11, (21, 31))
            for units in range(1, 41):
                gain = units / 10
                data = {}
                _, info = histogram_weighted_mapping_native_u8(
                    source, maximum, gain_limit=gain, interval_data=data
                )
                with self.subTest(maximum=maximum, gain=gain):
                    steps = np.diff(data["limited_lut_float"])
                    self.assertTrue(np.all((steps >= 0) & (steps <= gain + 1e-10)))
                    rounded_steps = np.diff(data["final_lut"].astype(np.int64))
                    self.assertTrue(np.all((rounded_steps >= 0) & (rounded_steps <= np.ceil(gain))))
                    anchor = info["gain_anchor_dn"]
                    self.assertEqual(data["final_lut"][anchor], data["before_gain_lut"][anchor])

    def test_fractional_csv_keeps_both_unrounded_and_integer_spans(self):
        source = self.example - 7900
        params = tone_mapping_parameters("histogram_weighted", {"histogram_weighted_gain_limit": 0.5})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fractional.csv"
            save_histogram_weighted_intervals(path, source, 255, params)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertTrue(all(float(row["gain_limit_gray_per_dn"]) == 0.5 for row in rows))
        self.assertEqual([float(rows[i]["limited_output_span_before_rounding"]) for i in range(98, 102)], [0.5] * 4)
        self.assertEqual([int(rows[i]["final_output_span"]) for i in range(98, 102)], [1, 0, 1, 0])

    def test_global_and_identical_tiles_match(self):
        for gain in (0, 1, 2, 4):
            expected, _ = histogram_weighted_mapping_native_u8(
                self.example, 16383, gain_limit=gain
            )
            global_output, _ = histogram_weighted_grid_u8(self.example, 16383, gain_limit=gain)
            np.testing.assert_array_equal(global_output, expected)
            tiled, info = histogram_weighted_grid_u8(
                np.tile(self.example, (2, 2)), 16383, grid=2, gain_limit=gain
            )
            np.testing.assert_array_equal(tiled, np.tile(expected, (2, 2)))
            self.assertTrue(all(tile["gain_limit"] == gain for tile in info["tiles"]))

    def test_csv_matches_final_luts_and_preserves_weight_columns(self):
        source = np.random.default_rng(18).integers(98, 103, (13, 21))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "intervals.csv"
            for grid in (1, 2):
                params = tone_mapping_parameters("histogram_weighted", {
                    "histogram_weighted_gain_limit": 2, "histogram_weighted_grid": grid,
                })
                save_histogram_weighted_intervals(path, source, 255, params)
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), grid * grid * 255)
                for ty, tx, tile in weighted_tiles(source, grid):
                    selected = [row for row in rows if grid == 1 or (
                        int(row["tile_y"]) == ty and int(row["tile_x"]) == tx
                    )]
                    data = {}
                    histogram_weighted_mapping_native_u8(tile, 255, interval_data=data, gain_limit=2)
                    for dn, row in enumerate(selected):
                        self.assertEqual(int(row["final_output_start"]), data["final_lut"][dn])
                        self.assertEqual(int(row["final_output_end"]), data["final_lut"][dn + 1])
                        self.assertEqual(int(row["before_gain_output_start"]), data["before_gain_lut"][dn])
                        self.assertEqual(int(row["gain_limit_gray_per_dn"]), 2)
                        self.assertLessEqual(int(row["final_output_span"]), 2)
                    self.assertAlmostEqual(sum(float(row["adjusted_ratio"]) for row in selected), 1)
                    self.assertAlmostEqual(sum(float(row["adjusted_output_span"]) for row in selected), 255)
                    self.assertEqual(sum(int(row["final_output_span"]) for row in selected),
                                     int(data["final_lut"][-1]) - int(data["final_lut"][0]))


if __name__ == "__main__":
    unittest.main()
