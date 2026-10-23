"""Effective-range allocation, plateau bounds, reports and Tone Mapping UI."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermal_toolkit.algorithms import weighted_aci
from thermal_toolkit.algorithms.parameters import DEFAULT_TONE_MAPPING_PARAMETERS
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping
from thermal_toolkit.algorithms.weighted_plateau import effective_weighted_plateau_u8
from thermal_toolkit.reports.weighted_plateau import save_weighted_plateau_intervals


class WeightedPlateauTests(unittest.TestCase):
    def setUp(self):
        self.source = np.repeat([100, 101, 102, 103], [100, 5, 5, 1]).reshape(3, 37)

    def test_exact_weighted_and_double_plateau_allocation(self):
        for mode, expected in (("weighted", [0, 172, 224, 255]),
                               ("double_plateau", [0, 109, 182, 255])):
            data = {}
            result, info = effective_weighted_plateau_u8(
                self.source, 16383, plateau_mode=mode, plateau_lower=0.5, interval_data=data)
            self.assertEqual(info['upper_plateau_count'], 28)
            self.assertEqual(info['lower_plateau_count'], 14 if mode == 'double_plateau' else 0)
            self.assertEqual(data['lut'][100:104].tolist(), expected)
            self.assertEqual(info['clipped_histogram_count'], 72)
            self.assertEqual(info['floor_added_histogram_count'], 31 if mode == 'double_plateau' else 0)
            np.testing.assert_array_equal(result, data['lut'][self.source])

    def test_no_allocation_outside_effective_range_and_offset_invariance(self):
        for redistribute in (False, True):
            data = {}
            result, info = effective_weighted_plateau_u8(
                self.source, 255, plateau_redistribute=redistribute, interval_data=data)
            shifted, _ = effective_weighted_plateau_u8(
                self.source + 8000, 16383, plateau_redistribute=redistribute)
            np.testing.assert_array_equal(result, shifted)
            self.assertEqual((info['output_min'], info['output_max']), (0, 255))
            self.assertFalse(np.any(data['weights'][:100]))
            self.assertFalse(np.any(data['weights'][103:]))

    def test_redistribution_and_floor_preserve_empty_bins_and_cap(self):
        source = np.repeat([100, 102, 104], [100, 5, 1]).reshape(2, 53)
        for upper in (0.1, 1, 2, 100):
            for redistribute in (False, True):
                data = {}
                _, info = effective_weighted_plateau_u8(
                    source, 255, plateau_upper=upper, plateau_lower=min(0.5, upper),
                    plateau_redistribute=redistribute, interval_data=data)
                counts = data['processed_histogram']
                self.assertEqual(counts[101], 0)
                self.assertEqual(counts[103], 0)
                self.assertTrue(np.all(counts <= info['upper_plateau_count']))
                self.assertTrue(np.all(counts[[100, 102, 104]] >= info['lower_plateau_count']))
                self.assertEqual(int(counts.sum()), source.size - info['discarded_histogram_count']
                                 + info['floor_added_histogram_count'])

    def test_redistribution_retains_counts_when_capacity_is_sufficient(self):
        _, info = effective_weighted_plateau_u8(
            self.source, 255, plateau_upper=1.5, plateau_lower=0,
            plateau_redistribute=True)
        self.assertEqual(info['discarded_histogram_count'], 0)
        self.assertEqual(info['processed_histogram_total'], self.source.size)

    def test_percentile_range_clamps_tails_before_histogram(self):
        source = np.array([0] + [100] * 50 + [101] * 50 + [16383]).reshape(6, 17)
        result, info = effective_weighted_plateau_u8(
            source, 16383, plateau_low_percentile=1, plateau_high_percentile=99)
        self.assertEqual((info['mapping_first_dn'], info['mapping_last_dn']), (100, 101))
        self.assertEqual((info['percentile_clipped_low'], info['percentile_clipped_high']), (1, 1))
        self.assertEqual(result.flat[0], 0)
        self.assertEqual(result.flat[-1], 255)
        self.assertEqual(info['unique_levels'], 2)

    def test_constant_and_collapsed_percentile_ranges_are_neutral(self):
        for value in (0, 8000, 16383):
            result, info = effective_weighted_plateau_u8(np.full((3, 7), value), 16383)
            self.assertTrue(np.all(result == 128))
            self.assertTrue(info['constant_effective_range'])
        source = np.full((10, 10), 8000)
        source[0, :2] = [0, 16383]
        result, _ = effective_weighted_plateau_u8(
            source, 16383, plateau_low_percentile=2, plateau_high_percentile=98)
        self.assertTrue(np.all(result == 128))

    def test_monotonic_lut_and_plateau_bounds_in_supported_domains(self):
        rng = np.random.default_rng(23)
        for maximum in (255, 16383, 65535):
            source = rng.integers(0, maximum + 1, (21, 31), dtype=np.uint16)
            for mode in ('weighted', 'double_plateau'):
                data = {}
                result, info = effective_weighted_plateau_u8(
                    source, maximum, plateau_mode=mode, plateau_redistribute=True, interval_data=data)
                self.assertTrue(np.all(np.diff(data['lut'].astype(int)) >= 0))
                self.assertEqual(result.dtype, np.uint8)
                self.assertTrue(result.flags.c_contiguous)
                self.assertEqual(result.shape, source.shape)
                self.assertLessEqual(int(data['processed_histogram'].max()), info['upper_plateau_count'])

    def test_zero_lower_matches_upper_only_and_aci_is_independent(self):
        previous = dict(weighted_aci.ACTIVE_WEIGHTED_ACI)
        try:
            first, _, _, parameters = apply_tone_mapping(
                self.source, 'weighted_plateau', 16383, {'plateau_lower': 0})
            for enabled in (False, True):
                weighted_aci.configure_weighted_aci(dict(enabled=enabled))
                second, _, info, _ = apply_tone_mapping(
                    self.source, 'weighted_plateau', 16383, {'plateau_mode': 'weighted'})
                np.testing.assert_array_equal(first, second)
                self.assertNotIn('aci_ratio', info)
            # A saved parameter snapshot reproduces the result without application Settings.
            third, _, _, _ = apply_tone_mapping(
                self.source, 'weighted_plateau', 16383, json.loads(json.dumps(parameters)))
            np.testing.assert_array_equal(first, third)
        finally:
            weighted_aci.configure_weighted_aci(previous)

    def test_invalid_parameters(self):
        for params in (
            {'plateau_mode': 'unknown'}, {'plateau_upper': 0}, {'plateau_upper': float('nan')},
            {'plateau_lower': 2}, {'plateau_upper': 0.1, 'plateau_lower': 0.5},
            {'plateau_low_percentile': 100}, {'plateau_high_percentile': float('nan')},
            {'plateau_gain_limit': -0.1}, {'plateau_gain_limit': 4.1},
            {'plateau_gain_limit': 0.25}, {'plateau_gain_limit': float('nan')},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                effective_weighted_plateau_u8(self.source, 16383, **params)
        with self.assertRaises(ValueError):
            effective_weighted_plateau_u8(np.empty((0, 0)), 16383)

    def test_csv_matches_lut_and_allocates_only_effective_intervals(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'], plateau_redistribute=True)
        data = {}
        effective_weighted_plateau_u8(self.source, 255, interval_data=data, **params)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'intervals.csv'
            save_weighted_plateau_intervals(destination, self.source, 255, params)
            with destination.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 255)
        self.assertAlmostEqual(sum(float(row['interval_ratio']) for row in rows), 1)
        for i, row in enumerate(rows):
            self.assertEqual(int(row['output_start']), int(data['lut'][i]))
            self.assertEqual(int(row['output_end']), int(data['lut'][i + 1]))
            if not 100 <= i < 103:
                self.assertEqual(float(row['interval_ratio']), 0)

    def test_fractional_gain_bounds_anchor_and_histogram_invariance(self):
        source = np.repeat(np.arange(100, 105), [100, 300, 500, 300, 100]).reshape(10, 130)
        original = {}
        effective_weighted_plateau_u8(source, 16383, interval_data=original)
        for gain in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
            data = {}
            result, info = effective_weighted_plateau_u8(
                source, 16383, plateau_gain_limit=gain, interval_data=data)
            self.assertEqual(info['gain_anchor_dn'], 102)
            self.assertEqual(data['lut'][102], original['lut'][102])
            steps = np.diff(data['limited_lut_float'])
            self.assertTrue(np.all(steps >= -1e-12))
            self.assertTrue(np.all(steps <= gain + 1e-12))
            np.testing.assert_array_equal(result, data['lut'][source])
            for key in ('histogram', 'processed_histogram', 'weights'):
                np.testing.assert_array_equal(data[key], original[key])
            if gain == 1:
                self.assertEqual(data['lut'][100:105].tolist(), [126, 127, 128, 129, 130])
            if gain == 0.5:
                self.assertEqual(data['lut'][100:105].tolist(), [127, 128, 128, 129, 129])
            self.assertTrue(np.all(data['lut'][:101] == data['lut'][100]))
            self.assertTrue(np.all(data['lut'][104:] == data['lut'][104]))

    def test_legacy_parameters_default_to_off_and_constant_is_unchanged(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'])
        del params['plateau_gain_limit']
        expected, _ = effective_weighted_plateau_u8(self.source, 16383)
        result, _, info, used = apply_tone_mapping(self.source, 'weighted_plateau', 16383, params)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(used['plateau_gain_limit'], 0)
        self.assertFalse(info['gain_limit_applied'])
        # This is an already populated DN after percentile clamping, including a
        # collapsed effective range with outliers in the original image.
        source = np.full((10, 10), 8000)
        source[0, :2] = [0, 16383]
        result, info = effective_weighted_plateau_u8(
            source, 16383, plateau_low_percentile=2, plateau_high_percentile=98,
            plateau_gain_limit=0.2)
        self.assertTrue(np.all(result == 128))
        self.assertEqual(info['gain_anchor_dn'], 8000)
        self.assertFalse(info['gain_limit_applied'])

    def test_gain_csv_matches_both_luts_and_fractional_steps(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'], plateau_gain_limit=0.2)
        data = {}
        effective_weighted_plateau_u8(self.source, 255, interval_data=data, **params)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'gain.csv'
            save_weighted_plateau_intervals(destination, self.source, 255, params)
            with destination.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows):
            self.assertEqual(float(row['gain_limit_gray_per_dn']), 0.2)
            self.assertEqual(int(row['before_gain_output_start']), int(data['before_gain_lut'][i]))
            self.assertEqual(int(row['output_start']), int(data['lut'][i]))
            self.assertAlmostEqual(float(row['limited_output_start_before_rounding']),
                                   data['limited_lut_float'][i])
            self.assertLessEqual(float(row['limited_output_span_before_rounding']), 0.2)


class WeightedPlateauUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from thermal_toolkit.core.qt import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_method_controls_reset_preview_and_parameter_snapshot(self):
        from thermal_toolkit.ui.main_window import ThermalRawToolkit
        with patch.object(ThermalRawToolkit, 'refresh_camera_devices'):
            window = ThermalRawToolkit()
        try:
            combo = window.raw_input_mapping_combo
            combo.setCurrentIndex(combo.findData('weighted_plateau'))
            window.sync_image_input_mapping()
            defaults = DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau']
            self.assertEqual(window.current_tone_mapping_parameters(), defaults)
            self.assertFalse(window.tone_plateau_upper.keyboardTracking())
            self.assertFalse(window.tone_plateau_gain_limit.keyboardTracking())
            window.tone_plateau_mode.setCurrentIndex(0)
            self.assertFalse(window.tone_mapping_parameter_rows['plateau_lower'].isEnabled())
            window.tone_plateau_upper.setValue(2)
            window.tone_plateau_redistribute.setChecked(True)
            window.tone_plateau_gain_limit.setValue(0.2)
            combo.setCurrentIndex(combo.findData('percentile'))
            combo.setCurrentIndex(combo.findData('weighted_plateau'))
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_upper'], 2)
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_gain_limit'], 0.2)
            window.reset_tone_mapping_parameters()
            self.assertEqual(window.current_tone_mapping_parameters(), defaults)
            self.assertTrue(window.tone_mapping_parameter_rows['plateau_lower'].isEnabled())
            window.raw_algorithm_info = dict(
                bit_depth=14, value_max=16383, value_name='RAW DN', source_type='raw')
            source = np.arange(8000, 8035).reshape(5, 7)
            with patch.object(window, 'make_output', side_effect=AssertionError('Preview wrote output')):
                result, maximum, _ = window.prepare_raw_pipeline_input(source)
            expected, _, info, _ = apply_tone_mapping(source, 'weighted_plateau', 16383)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(maximum, 255)
            self.assertEqual(window.raw_tone_mapping_parameters, defaults)
            self.assertEqual(window.raw_tone_mapping_calculated['mapping_first_dn'], info['mapping_first_dn'])
            self.assertIn('Upper=', window.tone_mapping_calculated.text())
            window.tone_plateau_gain_limit.setValue(0.2)
            result, _, _ = window.prepare_raw_pipeline_input(source)
            expected, _, _, _ = apply_tone_mapping(
                source, 'weighted_plateau', 16383, {'plateau_gain_limit': 0.2})
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(window.raw_tone_mapping_parameters['plateau_gain_limit'], 0.2)
            self.assertIn('Gain Limit=0.2', window.tone_mapping_calculated.text())
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
