"""Save-only plateau histogram and interval allocation report."""

import csv
import numpy as np

from thermal_toolkit.algorithms.weighted_plateau import weighted_plateau_u8
from thermal_toolkit.algorithms.tone_mapping import tone_mapping_parameters


def save_weighted_plateau_intervals(path, image, input_max, parameters):
    parameters = tone_mapping_parameters("weighted_plateau", parameters, input_max)
    data = {}
    _, info = weighted_plateau_u8(image, input_max, interval_data=data, **parameters)
    first, last = info["mapping_first_dn"], info["mapping_last_dn"]
    total = info["total_interval_weight"]
    before_total = info["before_bright_total_interval_weight"]
    source_percentiles = np.cumsum(data['source_histogram']) * (100.0 / image.size)
    with open(path, "w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "dn_start", "dn_end", "inside_input_range", "input_count_start",
            "input_count_end", "plateau_count_start", "plateau_count_end",
            "upper_plateau_count", "lower_plateau_count", "interval_weight", "interval_ratio",
            "output_start", "output_end", "output_span",
            "gain_limit_gray_per_dn", "before_gain_output_start", "before_gain_output_end",
            "before_gain_output_span", "limited_output_start_before_rounding",
            "limited_output_end_before_rounding", "limited_output_span_before_rounding",
            "bright_percentile", "bright_threshold_dn", "bright_multiplier", "dn_start_percentile",
            "before_bright_weight", "before_bright_ratio", "adjusted_weight", "adjusted_ratio",
            "before_bright_output_span", "adjusted_output_span",
            "weight_mode", "weight_formula",
        ])
        for i, weight in enumerate(data["weights"]):
            start, end = (int(v) for v in data["lut"][i:i + 2])
            before_start, before_end = (int(v) for v in data["before_gain_lut"][i:i + 2])
            limited_start, limited_end = (float(v) for v in data["limited_lut_float"][i:i + 2])
            before_weight = int(data['before_bright_weights'][i])
            before_ratio = before_weight / before_total if before_total else 0.0
            ratio = float(weight / total) if total else 0.0
            multiplier = (info['bright_weight'] if first <= i < last
                          and i + 0.5 >= info['bright_threshold_dn'] else 1.0)
            writer.writerow([
                i, i + 1, first <= i < last,
                *(int(v) for v in data["histogram"][i:i + 2]),
                *(int(v) for v in data["processed_histogram"][i:i + 2]),
                info["upper_plateau_count"], info["lower_plateau_count"], float(weight),
                ratio, start, end, end - start,
                info['gain_limit'], before_start, before_end, before_end - before_start,
                round(limited_start, 1), round(limited_end, 1), round(limited_end - limited_start, 1),
                info['bright_percentile'], info['bright_threshold_dn'], multiplier,
                float(source_percentiles[i]), before_weight, float(before_ratio),
                float(weight), ratio, before_ratio * 255, ratio * 255,
                info['weight_mode'], info['weight_formula'],
            ])
