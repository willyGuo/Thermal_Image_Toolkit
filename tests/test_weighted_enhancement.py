"""Check 8-bit Weighted Mapping stage integration and parameter propagation."""

import json
import unittest
import numpy as np

from thermal_toolkit.algorithms.parameters import AlgorithmStage, default_algorithm_parameters
from thermal_toolkit.algorithms.pipeline import apply_algorithm_pipeline, pipeline_execution_json
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.uvc.processing import process_uvc_pipeline_frame


class WeightedEnhancementTests(unittest.TestCase):
    def test_mapping_consumes_preceding_step_and_passes_all_parameters(self):
        raw = np.random.default_rng(42).integers(3000, 13000, (31, 43), dtype=np.uint16)
        tone, *_ = apply_tone_mapping(raw, "linear", 16383)
        gamma = AlgorithmStage("gamma", parameters={"gamma": 1.2})
        previous, _ = apply_algorithm_pipeline(tone, [gamma])
        for grid in (1, 3):
            for gain in (0, 0.1, 0.5, 1, 1.5, 2, 4):
                params = {
                    "histogram_weighted_grid": grid,
                    "histogram_weighted_clip": 0.7,
                    "histogram_weighted_redistribute": False,
                    "histogram_weighted_bright_percentile": 60.0,
                    "histogram_weighted_bright_weight": 1.7,
                    "histogram_weighted_gain_limit": gain,
                }
                with self.subTest(grid=grid, gain=gain):
                    stage = AlgorithmStage("histogram_weighted", parameters=params)
                    expected, *_ = apply_tone_mapping(previous, "histogram_weighted", 255, params)
                    result, records = apply_algorithm_pipeline(tone, [gamma, stage])
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_array_equal(records[0]["image"], previous)
                    self.assertEqual(result.dtype, np.uint8)
                    record = json.loads(json.dumps(pipeline_execution_json(records)))[1]
                    self.assertEqual(record["parameters"], params)
                    self.assertIn("8-bit", record["details"][0])

    def test_defaults_and_roundtrip_do_not_share_parameters(self):
        first, second = AlgorithmStage("histogram_weighted"), AlgorithmStage("histogram_weighted")
        first.parameters["histogram_weighted_gain_limit"] = 2
        self.assertEqual(second.parameters, default_algorithm_parameters("histogram_weighted"))
        self.assertEqual(second.parameters["histogram_weighted_gain_limit"], 0)
        saved = json.loads(json.dumps(first.to_dict()))
        restored = AlgorithmStage(saved["algorithm"], saved["enabled"], saved["parameters"])
        self.assertEqual(restored.parameters, first.parameters)

    def test_uvc_uses_same_eight_bit_mapping(self):
        raw = np.random.default_rng(2).integers(2000, 12000, (32, 48), dtype=np.uint16)
        stage = AlgorithmStage("histogram_weighted", parameters={"histogram_weighted_gain_limit": 2})
        for frame in (raw, raw_to_display_u8(raw, 16383)):
            with self.subTest(dtype=frame.dtype):
                payload = process_uvc_pipeline_frame(frame, {"stages": [stage], "raw_max": 16383, "revision": 1})
                expected_input = raw_to_display_u8(raw, 16383)
                np.testing.assert_array_equal(payload["pipeline_input"], expected_input)
                expected, _ = apply_algorithm_pipeline(expected_input, [stage])
                np.testing.assert_array_equal(payload["final_image"], expected)


if __name__ == "__main__":
    unittest.main()
