import unittest
import numpy as np

from thermal_toolkit.algorithms.mapping import percentile_preview_u8, percentile_stretch_gray, limit_percentile_gain
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping
from thermal_toolkit.algorithms.parameters import AlgorithmStage
from thermal_toolkit.algorithms.pipeline import apply_algorithm_pipeline
from thermal_toolkit.uvc.processing import process_uvc_pipeline_frame


class PercentileGainTests(unittest.TestCase):
    def test_known_narrow_range_and_fraction(self):
        source = np.arange(7998, 8003).reshape(1, 5)
        for gain, expected in ((0, [0, 63, 127, 191, 255]), (0.5, [126, 127, 127, 128, 128]),
                               (1, [125, 126, 127, 128, 129]), (2, [123, 125, 127, 129, 131])):
            output, _, info, params = apply_tone_mapping(source, "percentile", 16383, {
                "low_percentile": 0, "high_percentile": 100, "percentile_gain_limit": gain,
            })
            self.assertEqual(output.ravel().tolist(), expected)
            self.assertEqual(params["percentile_gain_limit"], gain)
            if gain:
                self.assertEqual(info["gain_anchor_dn"], 8000)
                self.assertEqual(info["effective_gain"], gain)

    def test_off_matches_existing_functions(self):
        rng = np.random.default_rng(4)
        for maximum in (255, 16383, 65535):
            for source in (rng.integers(0, maximum + 1, (31, 43)), np.full((15, 15), 100)):
                for low, high in ((0, 100), (1, 99), (20, 80)):
                    expected, *_ = percentile_preview_u8(source, low, high)
                    actual, *_ = apply_tone_mapping(source, "percentile", maximum, {"low_percentile": low, "high_percentile": high})
                    np.testing.assert_array_equal(actual, expected)
                    if maximum == 255:
                        expected, *_ = percentile_stretch_gray(source.astype(np.uint8), low, high)
                        actual, _ = apply_algorithm_pipeline(source.astype(np.uint8), [AlgorithmStage("auto_contrast", parameters={"low_percentile": low, "high_percentile": high})])
                        np.testing.assert_array_equal(actual, expected)

    def test_inactive_limit_is_bit_exact(self):
        source = np.arange(1024).reshape(32, 32)
        expected, *_ = percentile_preview_u8(source)
        output, _, info, _ = apply_tone_mapping(source, "percentile", 16383, {"percentile_gain_limit": 0.5})
        self.assertFalse(info["gain_limit_applied"])
        np.testing.assert_array_equal(output, expected)

    def test_percentile_clipping_monotonicity_and_rounding(self):
        source = np.arange(256).reshape(1, 256)
        for units in range(1, 41):
            gain = units / 10
            output, _, info, _ = apply_tone_mapping(source, "percentile", 255, {
                "low_percentile": 40, "high_percentile": 60, "percentile_gain_limit": gain,
            })
            steps = np.diff(output.astype(np.int64))
            self.assertTrue(np.all((steps >= 0) & (steps <= np.ceil(gain))))
            self.assertTrue(np.all(output[source <= info["low_dn"]] == output[0, 0]))
            self.assertTrue(np.all(output[source >= info["high_dn"]] == output[0, -1]))

    def test_uvc_and_enhancement_share_fractional_limit(self):
        source = np.tile(np.arange(98, 103, dtype=np.uint8), (10, 4))
        stage = AlgorithmStage("auto_contrast", parameters={"percentile_gain_limit": 0.5})
        output, records = apply_algorithm_pipeline(source, [stage])
        original, low, high = percentile_stretch_gray(source)
        expected, _ = limit_percentile_gain(source, original, low, high, 0.5)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(records[0]["parameters"]["percentile_gain_limit"], 0.5)
        payload = process_uvc_pipeline_frame(source, {"stages": [stage], "raw_max": 16383, "revision": 1})
        np.testing.assert_array_equal(payload["final_image"], expected)

    def test_invalid_and_constant(self):
        source = np.full((8, 8), 100, dtype=np.uint8)
        for gain in (-1, 0.05, 4.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                apply_tone_mapping(source, "percentile", 255, {"percentile_gain_limit": gain})
        for gain in (0.1, 0.5, 1, 4):
            output, *_ = apply_tone_mapping(source, "percentile", 255, {"percentile_gain_limit": gain})
            self.assertFalse(output.any())
            output, _ = apply_algorithm_pipeline(source, [AlgorithmStage("auto_contrast", parameters={"percentile_gain_limit": gain})])
            np.testing.assert_array_equal(output, source)


if __name__ == "__main__":
    unittest.main()
