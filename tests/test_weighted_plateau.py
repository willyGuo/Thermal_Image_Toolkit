"""Observed input range allocation, plateau bounds, reports and Tone Mapping UI."""

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
from thermal_toolkit.algorithms.weighted_plateau import weighted_plateau_u8
from thermal_toolkit.reports.weighted_plateau import save_weighted_plateau_intervals


class WeightedPlateauTests(unittest.TestCase):
    def setUp(self):
        self.source = np.repeat([100, 101, 102, 103], [100, 5, 5, 1]).reshape(3, 37)

    def test_exact_double_plateau_allocation(self):
        data = {}
        result, info = weighted_plateau_u8(
            self.source, 16383, plateau_lower=0.5, interval_data=data)
        self.assertEqual(info['upper_plateau_count'], 28)
        self.assertEqual(info['lower_plateau_count'], 14)
        self.assertEqual(data['lut'][100:104].tolist(), [0, 109, 182, 255])
        self.assertEqual(info['clipped_histogram_count'], 72)
        self.assertEqual(info['floor_added_histogram_count'], 31)
        np.testing.assert_array_equal(result, data['lut'][self.source])

    def test_single_bin_weights_and_endpoint_normalization(self):
        data = {}
        result, info = weighted_plateau_u8(
            self.source, 16383, plateau_lower=0.5, plateau_weight_mode='single', interval_data=data)
        self.assertEqual(info['weight_mode'], 'single')
        self.assertEqual(info['weight_formula'], "H'(i)")
        self.assertEqual(data['before_bright_weights'][100:103].tolist(), [28, 14, 14])
        self.assertEqual(data['lut'][100:104].tolist(), [0, 128, 191, 255])
        self.assertEqual(info['total_interval_weight'], 56)
        np.testing.assert_array_equal(result, data['lut'][self.source])
        self.assertFalse(np.any(data['weights'][:100]))
        self.assertFalse(np.any(data['weights'][103:]))

    def test_weight_mode_preserves_plateaus_and_gain_follows_bright_weight(self):
        for redistribute in (False, True):
            adjacent, single = {}, {}
            options = dict(plateau_redistribute=redistribute,
                           plateau_bright_weight=1.5, plateau_bright_percentile=0,
                           plateau_gain_limit=0.2)
            weighted_plateau_u8(self.source, 16383, interval_data=adjacent, **options)
            _, info = weighted_plateau_u8(
                self.source, 16383, plateau_weight_mode='single', interval_data=single, **options)
            for key in ('histogram', 'processed_histogram'):
                np.testing.assert_array_equal(adjacent[key], single[key])
            np.testing.assert_array_equal(single['before_bright_weights'][100:103],
                                          single['processed_histogram'][100:103])
            np.testing.assert_allclose(single['weights'], single['before_bright_weights'] * 1.5)
            self.assertTrue(np.all(np.diff(single['limited_lut_float']) <= 0.2 + 1e-12))
            self.assertEqual(single['lut'][info['gain_anchor_dn']], info['gain_anchor_gray'])
        constant, _ = weighted_plateau_u8(
            np.full((3, 7), 8000), 16383, plateau_weight_mode='single')
        self.assertTrue(np.all(constant == 128))

    def test_no_allocation_outside_input_range_and_offset_invariance(self):
        for redistribute in (False, True):
            data = {}
            result, info = weighted_plateau_u8(
                self.source, 255, plateau_redistribute=redistribute, interval_data=data)
            shifted, _ = weighted_plateau_u8(
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
                _, info = weighted_plateau_u8(
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
        _, info = weighted_plateau_u8(
            self.source, 255, plateau_upper=1.5, plateau_lower=0,
            plateau_redistribute=True)
        self.assertEqual(info['discarded_histogram_count'], 0)
        self.assertEqual(info['processed_histogram_total'], self.source.size)

    def test_full_input_histogram_keeps_tails_and_ignores_old_range_settings(self):
        source = np.array([0] + [100] * 50 + [101] * 50 + [16383]).reshape(6, 17)
        data = {}
        result, info = weighted_plateau_u8(source, 16383, interval_data=data)
        self.assertEqual((info['mapping_first_dn'], info['mapping_last_dn']), (0, 16383))
        self.assertEqual(info['unique_levels'], 4)
        np.testing.assert_array_equal(data['histogram'], np.bincount(source.ravel(), minlength=16384))
        options = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'], plateau_gain_limit=0,
                       plateau_lower=0.1, plateau_low_percentile=1, plateau_high_percentile=99,
                       plateau_mode='weighted')
        restored, _, _, used = apply_tone_mapping(source, 'weighted_plateau', 16383, options)
        np.testing.assert_array_equal(result, restored)
        self.assertNotIn('plateau_mode', used)
        self.assertNotIn('plateau_low_percentile', used)
        self.assertNotIn('plateau_high_percentile', used)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'legacy.csv'
            save_weighted_plateau_intervals(destination, source, 16383, options)
            with destination.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(int(rows[0]['input_count_start']), 1)
        self.assertNotIn('inside_effective_range', rows[0])
        self.assertEqual(int(rows[-1]['input_count_end']), 1)

    def test_constant_input_is_neutral(self):
        for value in (0, 8000, 16383):
            result, info = weighted_plateau_u8(np.full((3, 7), value), 16383)
            self.assertTrue(np.all(result == 128))
            self.assertTrue(info['constant_input_range'])

    def test_monotonic_lut_and_plateau_bounds_in_supported_domains(self):
        rng = np.random.default_rng(23)
        for maximum in (255, 16383, 65535):
            source = rng.integers(0, maximum + 1, (21, 31), dtype=np.uint16)
            data = {}
            result, info = weighted_plateau_u8(
                source, maximum, plateau_redistribute=True, interval_data=data)
            self.assertTrue(np.all(np.diff(data['lut'].astype(int)) >= 0))
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(result.flags.c_contiguous)
            self.assertEqual(result.shape, source.shape)
            self.assertLessEqual(int(data['processed_histogram'].max()), info['upper_plateau_count'])

    def test_double_plateau_is_independent_of_aci(self):
        previous = dict(weighted_aci.ACTIVE_WEIGHTED_ACI)
        try:
            first, _, _, parameters = apply_tone_mapping(
                self.source, 'weighted_plateau', 16383)
            for enabled in (False, True):
                weighted_aci.configure_weighted_aci(dict(enabled=enabled))
                second, _, info, _ = apply_tone_mapping(
                    self.source, 'weighted_plateau', 16383)
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
            {'plateau_upper': 0}, {'plateau_upper': float('nan')},
            {'plateau_lower': 2}, {'plateau_upper': 0.1, 'plateau_lower': 0.5},
            {'plateau_gain_limit': -0.1}, {'plateau_gain_limit': 4.1},
            {'plateau_gain_limit': 0.25}, {'plateau_gain_limit': float('nan')},
            {'plateau_bright_weight': 0.9}, {'plateau_bright_weight': 1.15},
            {'plateau_bright_weight': float('inf')}, {'plateau_bright_percentile': -1},
            {'plateau_bright_percentile': 101}, {'plateau_bright_percentile': float('nan')},
            {'plateau_weight_mode': 'invalid'},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                weighted_plateau_u8(self.source, 16383, **params)
        with self.assertRaises(ValueError):
            weighted_plateau_u8(np.empty((0, 0)), 16383)

    def test_csv_matches_lut_and_allocates_only_input_intervals(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'], plateau_redistribute=True)
        data = {}
        weighted_plateau_u8(self.source, 255, interval_data=data, **params)
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
        weighted_plateau_u8(source, 16383, interval_data=original)
        for gain in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
            data = {}
            result, info = weighted_plateau_u8(
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

    def test_missing_parameters_use_tone_defaults_and_constant_is_unchanged(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'])
        del params['plateau_gain_limit']
        del params['plateau_bright_weight']
        del params['plateau_bright_percentile']
        del params['plateau_weight_mode']
        expected, _ = weighted_plateau_u8(
            self.source, 16383, **DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'])
        result, _, info, used = apply_tone_mapping(self.source, 'weighted_plateau', 16383, params)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(used['plateau_gain_limit'], 2)
        self.assertEqual(used['plateau_bright_weight'], 1)
        # Explicit parameters from older saved results retain their behavior.
        legacy = dict(plateau_upper=1, plateau_lower=0.1, plateau_gain_limit=0)
        expected, _ = weighted_plateau_u8(self.source, 16383, **legacy)
        result, _, _, _ = apply_tone_mapping(self.source, 'weighted_plateau', 16383, legacy)
        np.testing.assert_array_equal(result, expected)
        source = np.full((10, 10), 8000)
        result, info = weighted_plateau_u8(source, 16383, plateau_gain_limit=0.2)
        self.assertTrue(np.all(result == 128))
        self.assertEqual(info['gain_anchor_dn'], 8000)
        self.assertFalse(info['gain_limit_applied'])

    def test_gain_csv_matches_both_luts_and_fractional_steps(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'], plateau_gain_limit=0.2)
        data = {}
        weighted_plateau_u8(self.source, 255, interval_data=data, **params)
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

    def test_bright_weights_and_exact_normalized_mapping(self):
        source = np.tile(np.arange(100, 105), (10, 1))
        original = {}
        weighted_plateau_u8(source, 255, interval_data=original)
        data = {}
        _, info = weighted_plateau_u8(
            source, 255, plateau_bright_percentile=60,
            plateau_bright_weight=1.5, interval_data=data)
        self.assertAlmostEqual(info['bright_threshold_dn'], 102.4)
        self.assertEqual(data['weights'][100:104].tolist(), [20, 20, 30, 30])
        self.assertEqual(data['before_bright_weights'][100:104].tolist(), [20, 20, 20, 20])
        self.assertEqual(data['before_gain_lut'][100:105].tolist(), [0, 51, 102, 179, 255])
        for key in ('histogram', 'processed_histogram'):
            np.testing.assert_array_equal(data[key], original[key])
        self.assertFalse(np.any(data['weights'][:100]))
        self.assertFalse(np.any(data['weights'][104:]))

    def test_bright_threshold_uses_original_input_and_gain_is_last(self):
        source = np.arange(101).reshape(1, 101)
        data = {}
        _, info = weighted_plateau_u8(
            source, 255, plateau_bright_percentile=80,
            plateau_bright_weight=2, interval_data=data)
        self.assertEqual(info['bright_threshold_dn'], 80)
        self.assertEqual(info['mapping_last_dn'], 100)
        np.testing.assert_array_equal(data['weights'][:80], data['before_bright_weights'][:80])
        np.testing.assert_array_equal(data['weights'][80:100], data['before_bright_weights'][80:100] * 2)
        for percentile in (0, 60, 100):
            before, limited = {}, {}
            weighted_plateau_u8(source, 255, plateau_bright_weight=1.05,
                                         plateau_bright_percentile=percentile, interval_data=before)
            _, info = weighted_plateau_u8(
                source, 255, plateau_bright_weight=1.05, plateau_bright_percentile=percentile,
                plateau_gain_limit=0.2, interval_data=limited)
            np.testing.assert_array_equal(before['lut'], limited['before_gain_lut'])
            np.testing.assert_array_equal(before['weights'], limited['weights'])
            self.assertTrue(np.all(np.diff(limited['limited_lut_float']) <= 0.2 + 1e-12))
            self.assertEqual(limited['lut'][info['gain_anchor_dn']],
                             before['lut'][info['gain_anchor_dn']])
        constant, _ = weighted_plateau_u8(np.full((3, 7), 8000), 16383,
                                                  plateau_bright_weight=2, plateau_gain_limit=0.2)
        self.assertTrue(np.all(constant == 128))

    def test_bright_csv_preserves_fractional_weights_and_reports_source_percentiles(self):
        source = np.arange(100, 105).reshape(1, 5)
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'],
                      plateau_bright_weight=1.05, plateau_bright_percentile=60,
                      plateau_gain_limit=0.2)
        data = {}
        weighted_plateau_u8(source, 255, interval_data=data, **params)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'bright.csv'
            save_weighted_plateau_intervals(destination, source, 255, params)
            with destination.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(sum(float(row['before_bright_ratio']) for row in rows), 1)
        self.assertAlmostEqual(sum(float(row['adjusted_ratio']) for row in rows), 1)
        for i, row in enumerate(rows):
            self.assertEqual(float(row['interval_weight']), float(data['weights'][i]))
            self.assertEqual(float(row['adjusted_weight']), float(data['weights'][i]))
            self.assertEqual(float(row['before_bright_weight']), float(data['before_bright_weights'][i]))
            self.assertEqual(int(row['output_start']), int(data['lut'][i]))
            self.assertAlmostEqual(float(row['dn_start_percentile']), float(np.mean(source <= i) * 100))
        self.assertEqual(float(rows[102]['bright_multiplier']), 1.05)
        self.assertEqual(float(rows[101]['bright_multiplier']), 1)
        self.assertAlmostEqual(float(rows[102]['interval_weight']), 2.1)

    def test_single_bin_csv_and_saved_parameter_roundtrip(self):
        params = dict(DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau'],
                      plateau_weight_mode='single', plateau_gain_limit=0.2,
                      plateau_bright_weight=1.05, plateau_bright_percentile=0)
        data = {}
        expected, _ = weighted_plateau_u8(self.source, 255, interval_data=data, **params)
        result, details, _, used = apply_tone_mapping(self.source, 'weighted_plateau', 255, params)
        np.testing.assert_array_equal(result, expected)
        self.assertIn("W(i)=H'(i);", details)
        replay, _, _, _ = apply_tone_mapping(
            self.source, 'weighted_plateau', 255, json.loads(json.dumps(used)))
        np.testing.assert_array_equal(replay, expected)
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'single.csv'
            save_weighted_plateau_intervals(destination, self.source, 255, params)
            with destination.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(sum(float(row['interval_ratio']) for row in rows), 1)
        for i, row in enumerate(rows):
            self.assertEqual(row['weight_mode'], 'single')
            self.assertEqual(row['weight_formula'], "H'(i)")
            self.assertEqual(float(row['interval_weight']), float(data['weights'][i]))
            self.assertEqual(int(row['output_start']), int(data['lut'][i]))


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
            self.assertFalse(hasattr(window, 'tone_plateau_low_percentile'))
            self.assertFalse(hasattr(window, 'tone_plateau_high_percentile'))
            defaults = DEFAULT_TONE_MAPPING_PARAMETERS['weighted_plateau']
            self.assertEqual(window.current_tone_mapping_parameters(), defaults)
            self.assertFalse(window.tone_plateau_upper.keyboardTracking())
            self.assertFalse(window.tone_plateau_gain_limit.keyboardTracking())
            self.assertFalse(window.tone_plateau_bright_percentile.keyboardTracking())
            self.assertFalse(window.tone_mapping_parameter_rows['plateau_bright_percentile'].isEnabled())
            self.assertFalse(hasattr(window, 'tone_plateau_mode'))
            window.tone_plateau_weight_mode.setCurrentIndex(window.tone_plateau_weight_mode.findData('single'))
            self.assertTrue(window.tone_mapping_parameter_rows['plateau_lower'].isEnabled())
            window.tone_plateau_upper.setValue(2)
            window.tone_plateau_redistribute.setChecked(True)
            window.tone_plateau_gain_limit.setValue(0.2)
            window.tone_plateau_bright_weight.setCurrentIndex(window.tone_plateau_bright_weight.findData(1.5))
            window.tone_plateau_bright_percentile.setValue(60)
            self.assertTrue(window.tone_mapping_parameter_rows['plateau_bright_percentile'].isEnabled())
            combo.setCurrentIndex(combo.findData('percentile'))
            combo.setCurrentIndex(combo.findData('weighted_plateau'))
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_upper'], 2)
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_gain_limit'], 0.2)
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_bright_weight'], 1.5)
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_bright_percentile'], 60)
            self.assertEqual(window.current_tone_mapping_parameters()['plateau_weight_mode'], 'single')
            window.reset_tone_mapping_parameters()
            self.assertEqual(window.current_tone_mapping_parameters(), defaults)
            self.assertFalse(window.tone_mapping_parameter_rows['plateau_bright_percentile'].isEnabled())
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
            window.tone_plateau_bright_weight.setCurrentIndex(window.tone_plateau_bright_weight.findData(1.5))
            window.tone_plateau_bright_percentile.setValue(60)
            result, _, _ = window.prepare_raw_pipeline_input(source)
            expected, _, _, _ = apply_tone_mapping(source, 'weighted_plateau', 16383, {
                'plateau_gain_limit': 0.2, 'plateau_bright_weight': 1.5, 'plateau_bright_percentile': 60})
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(window.raw_tone_mapping_parameters['plateau_bright_weight'], 1.5)
            self.assertIn('Bright Weight=1.5', window.tone_mapping_calculated.text())
            window.tone_plateau_weight_mode.setCurrentIndex(window.tone_plateau_weight_mode.findData('single'))
            result, _, _ = window.prepare_raw_pipeline_input(source)
            expected, _, _, _ = apply_tone_mapping(
                source, 'weighted_plateau', 16383, window.current_tone_mapping_parameters())
            np.testing.assert_array_equal(result, expected)
            self.assertIn("W(i)=H'(i);", window.tone_mapping_calculated.text())
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main()
