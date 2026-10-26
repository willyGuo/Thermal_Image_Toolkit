"""Automatic clipping boundaries, pipeline parity and Settings validation."""

import unittest
import numpy as np

from thermal_toolkit.algorithms import weighted_aci
from thermal_toolkit.algorithms.histogram import histogram_weighted_mapping_native_u8, _redistribute_histogram_with_cap
from thermal_toolkit.algorithms.weighted_grid import histogram_weighted_grid_u8
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping
from thermal_toolkit.core.settings import default_toolkit_settings, validated_toolkit_settings


class WeightedAciTests(unittest.TestCase):
    def setUp(self):
        self.previous = dict(weighted_aci.ACTIVE_WEIGHTED_ACI)
        weighted_aci.configure_weighted_aci(weighted_aci.default_weighted_aci_settings())

    def tearDown(self):
        weighted_aci.configure_weighted_aci(self.previous)

    def test_low_high_and_mixed_contrast(self):
        for values, expected in (([1000, 1001], 0), ([1000, 2000], 1)):
            source = np.tile(values, (100, 100))
            _, info = histogram_weighted_mapping_native_u8(source, 16383)
            self.assertEqual(info['aci_ratio'], expected)
            self.assertEqual(info['clip_count_per_bin'], 20 if expected == 0 else source.size)
        # A tiny bright object raises sigma while P1/P99 remain identical.
        source = np.full((100, 100), 1000)
        source[0, 0] = 16383
        _, info = histogram_weighted_mapping_native_u8(source, 16383)
        self.assertGreater(info['aci_sigma'], 121)
        self.assertEqual(info['aci_range'], 0)
        self.assertEqual(info['clip_count_per_bin'], 20)

    def test_middle_uses_smaller_ratio_and_actual_cap(self):
        source = np.tile([1000, 1200], (100, 100))
        _, info = histogram_weighted_mapping_native_u8(source, 16383, redistribute=False)
        ratio = min((200 - 75) / 275, (100 - 18) / 103)
        self.assertAlmostEqual(info['aci_ratio'], ratio)
        self.assertEqual(info['clip_count_per_bin'], round(20 + ratio * (source.size - 20)))
        # Changing the manual clip has no effect in automatic mode.
        a, _ = histogram_weighted_mapping_native_u8(source, 16383, clip_limit=0)
        b, _ = histogram_weighted_mapping_native_u8(source, 16383, clip_limit=100)
        np.testing.assert_array_equal(a, b)

    def test_ratio_combination_min_mean_max(self):
        source = np.tile([100, 300], (100, 100))
        for mode, expected in (('min', 0.2), ('mean', 0.5), ('max', 0.8)):
            weighted_aci.configure_weighted_aci(dict(
                ratio_mode=mode, range_thl=0, range_thh=1000,
                sigma_thl=20, sigma_thh=120))
            _, info = histogram_weighted_mapping_native_u8(source, 16383)
            self.assertAlmostEqual(info['aci_range_ratio'], 0.2)
            self.assertAlmostEqual(info['aci_sigma_ratio'], 0.8)
            self.assertAlmostEqual(info['aci_ratio'], expected)
            self.assertEqual(info['clip_count_per_bin'], round(20 + expected * (source.size - 20)))
            self.assertEqual(info['aci_settings']['ratio_mode'], mode)
        self.assertEqual(weighted_aci.validate_weighted_aci_settings({})['ratio_mode'], 'min')
        self.assertEqual(weighted_aci.validate_weighted_aci_settings({'ratio_mode': 'invalid'})['ratio_mode'], 'min')

    def test_single_parameter_boundaries_and_ratio_combination_ignored(self):
        source = np.tile([100, 300], (100, 100))  # spread=200, mean=200
        for low, high, expected in ((1, 2, 0), (0, 2, .5), (0, 1, 1)):
            for mode in ('min', 'mean', 'max'):
                weighted_aci.configure_weighted_aci(dict(
                    parameter_mode='single', relative_range_thl=low,
                    relative_range_thh=high, ratio_mode=mode))
                _, info = histogram_weighted_mapping_native_u8(source, 16383)
                self.assertEqual(info['aci_mean'], 200)
                self.assertEqual(info['aci_relative_range'], 1)
                self.assertEqual(info['aci_ratio'], expected)
                self.assertEqual(info['clip_count_per_bin'], round(20 + expected * (source.size - 20)))
                details = weighted_aci.describe_weighted_aci(info)
                self.assertIn('(P99-P1) / mean', details)
                self.assertNotIn('Standard deviation=', details)

    def test_single_parameter_scale_invariance_and_zero_mean(self):
        weighted_aci.configure_weighted_aci(dict(parameter_mode='single'))
        source = np.tile([10, 20, 30, 40], (50, 50)).astype(np.uint16)
        output, info = histogram_weighted_mapping_native_u8(source, 16383, redistribute=False)
        scaled, scaled_info = histogram_weighted_mapping_native_u8(source * 100, 16383, redistribute=False)
        self.assertAlmostEqual(info['aci_ratio'], scaled_info['aci_ratio'])
        np.testing.assert_array_equal(output, scaled)
        for value in (0, 1000):
            _, constant = histogram_weighted_mapping_native_u8(np.full((20, 20), value), 16383)
            self.assertEqual(constant['aci_relative_range'], 0)
            self.assertEqual(constant['aci_ratio'], 0)
            self.assertTrue(np.isfinite(constant['aci_clip_count']))

    def test_single_parameter_grid_uses_tile_mean_and_details(self):
        from thermal_toolkit.ui.weighted_mapping_details import weighted_mapping_details_html
        weighted_aci.configure_weighted_aci(dict(parameter_mode='single', relative_range_thh=2))
        tile = np.tile([100, 300], (20, 10))
        source = np.block([[tile, tile + 1000], [tile + 2000, tile + 3000]])
        actual, details, calculated, parameters = apply_tone_mapping(
            source, 'histogram_weighted', 16383, {'histogram_weighted_grid': 2})
        expected, info = histogram_weighted_grid_u8(
            source, 16383, grid=2, gain_limit=parameters['histogram_weighted_gain_limit'])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(calculated['tiles'], info['tiles'])
        for tile_info, mean in zip(info['tiles'], (200, 1200, 2200, 3200)):
            self.assertEqual(tile_info['aci_mean'], mean)
            self.assertAlmostEqual(tile_info['aci_ratio'], 100 / mean)
        html = weighted_mapping_details_html(
            dict(bit_depth=14, value_name='DN', value_max=16383), calculated, parameters)
        self.assertIn('Single parameter', html)
        self.assertIn('(P99-P1) / mean', html)
        self.assertNotIn('Standard deviation', html)

    def test_single_parameter_settings_and_legacy_defaults(self):
        import json
        settings = default_toolkit_settings()
        settings['weighted_mapping'].update(parameter_mode='single', relative_range_thl=.0123, relative_range_thh=.4567)
        self.assertEqual(validated_toolkit_settings(json.loads(json.dumps(settings))), settings)
        for mode in (None, 'invalid', 1):
            self.assertEqual(weighted_aci.validate_weighted_aci_settings(dict(parameter_mode=mode))['parameter_mode'], 'dual')
        for value in (float('nan'), float('inf'), -1, 'invalid', 1):
            result = weighted_aci.validate_weighted_aci_settings(dict(relative_range_thl=value, relative_range_thh=1))
            self.assertEqual((result['relative_range_thl'], result['relative_range_thh']), (0, 1))
        legacy = dict(range_thl=80, range_thh=400, sigma_thl=20, sigma_thh=150, ratio_mode='max')
        weighted_aci.configure_weighted_aci(legacy)
        info = weighted_aci.weighted_aci_clip(np.tile([100, 300], (10, 10)), 2)
        self.assertEqual(info['aci_settings']['parameter_mode'], 'dual')
        self.assertAlmostEqual(info['aci_ratio'], max((200 - 80) / 320, (100 - 20) / 130))

    def test_grid_and_tone_mapping_share_automatic_clip(self):
        source = np.tile(np.arange(1000, 1400, dtype=np.uint16), (10, 1))
        output, info = histogram_weighted_grid_u8(source, 16383, grid=2, gain_limit=2)
        actual, details, calculated, _ = apply_tone_mapping(
            source, 'histogram_weighted', 16383, {'histogram_weighted_grid': 2})
        np.testing.assert_array_equal(actual, output)
        self.assertTrue(all('aci_ratio' in tile for tile in info['tiles']))
        self.assertIn('Automatic Clip', details)
        self.assertEqual(calculated['tiles'], info['tiles'])

    def test_manual_constant_and_small_image(self):
        weighted_aci.configure_weighted_aci(dict(enabled=False))
        _, info = histogram_weighted_mapping_native_u8(np.tile([1000, 1001], (100, 100)), 16383)
        self.assertEqual(info['clip_count_per_bin'], 20000)
        weighted_aci.configure_weighted_aci({})
        output, _ = histogram_weighted_mapping_native_u8(np.full((512, 640), 1000), 16383)
        self.assertTrue(np.all(output == round(1000 * 255 / 16383)))
        _, info = histogram_weighted_mapping_native_u8(np.array([[1000, 1001]]), 16383)
        self.assertEqual(info['clip_count_per_bin'], 2)

    def test_settings_invalid_thresholds_and_roundtrip(self):
        settings = default_toolkit_settings()
        settings['weighted_mapping'].update(range_thl=100, range_thh=500)
        self.assertEqual(validated_toolkit_settings(settings)['weighted_mapping'], settings['weighted_mapping'])
        for value in (float('nan'), float('inf'), 70000, -1, 'invalid'):
            settings['weighted_mapping'].update(range_thl=value, range_thh=10)
            result = validated_toolkit_settings(settings)['weighted_mapping']
            self.assertEqual((result['range_thl'], result['range_thh']), (75, 350))

    def test_raw14_clip20_is_full_scale_linear_at_every_dn(self):
        source = np.resize(np.arange(9980, 9997, dtype=np.uint16), (512, 640))
        data, lut_data = {}, {}
        output, info = histogram_weighted_mapping_native_u8(
            source, 16383, interval_data=data, lut_data=lut_data)
        expected_lut = np.rint(np.arange(16384) * 255.0 / 16383).astype(np.uint8)
        np.testing.assert_array_equal(lut_data['lut'], expected_lut)
        np.testing.assert_array_equal(output, expected_lut[source])
        self.assertEqual(info['clip_count_per_bin'], 20)
        self.assertEqual(info['discarded_histogram_count'], 0)
        self.assertEqual(info['redistribution_scope'], 'full_input_range')
        self.assertEqual((info['mapping_first_dn'], info['mapping_last_dn']), (0, 16383))
        self.assertTrue(np.all(data['before_bright_weights'] == 40))
        # A sub-gray span can straddle one uint8 rounding boundary.
        self.assertLessEqual(int(output.max()) - int(output.min()), 1)

    def test_redistribution_populates_empty_bins_and_preserves_cap(self):
        histogram = np.array([0, 0, 100, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(_redistribute_histogram_with_cap(histogram, 20),
                                      np.full(5, 20))
        for histogram in (np.array([0, 0, 10, 0, 0]), np.array([0, 0, 1000, 0, 0])):
            result = _redistribute_histogram_with_cap(histogram, 20)
            self.assertTrue(np.all(result <= 20))
            self.assertEqual(result.sum(), min(histogram.sum(), 100))

    def test_redistribution_off_retains_populated_range_mapping(self):
        source = np.resize(np.arange(9980, 9997, dtype=np.uint16), (512, 640))
        output, info = histogram_weighted_mapping_native_u8(source, 16383, redistribute=False)
        self.assertEqual((int(output.min()), int(output.max())), (0, 255))
        self.assertEqual(info['redistribution_scope'], 'none')


if __name__ == '__main__':
    unittest.main()
