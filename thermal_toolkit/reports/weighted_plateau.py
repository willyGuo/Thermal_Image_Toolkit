"""Save-only effective-range plateau histogram and interval allocation report."""

import csv

from thermal_toolkit.algorithms.weighted_plateau import effective_weighted_plateau_u8


def save_weighted_plateau_intervals(path, image, input_max, parameters):
    data = {}
    _, info = effective_weighted_plateau_u8(image, input_max, interval_data=data, **parameters)
    first, last = info["mapping_first_dn"], info["mapping_last_dn"]
    total = info["total_interval_weight"]
    with open(path, "w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "dn_start", "dn_end", "inside_effective_range", "input_count_start_after_range_clamp",
            "input_count_end_after_range_clamp", "plateau_count_start", "plateau_count_end",
            "upper_plateau_count", "lower_plateau_count", "interval_weight", "interval_ratio",
            "output_start", "output_end", "output_span",
            "gain_limit_gray_per_dn", "before_gain_output_start", "before_gain_output_end",
            "before_gain_output_span", "limited_output_start_before_rounding",
            "limited_output_end_before_rounding", "limited_output_span_before_rounding",
        ])
        for i, weight in enumerate(data["weights"]):
            start, end = (int(v) for v in data["lut"][i:i + 2])
            before_start, before_end = (int(v) for v in data["before_gain_lut"][i:i + 2])
            limited_start, limited_end = (float(v) for v in data["limited_lut_float"][i:i + 2])
            writer.writerow([
                i, i + 1, first <= i < last,
                *(int(v) for v in data["histogram"][i:i + 2]),
                *(int(v) for v in data["processed_histogram"][i:i + 2]),
                info["upper_plateau_count"], info["lower_plateau_count"], int(weight),
                float(weight / total) if total else 0.0, start, end, end - start,
                info['gain_limit'], before_start, before_end, before_end - before_start,
                round(limited_start, 1), round(limited_end, 1), round(limited_end - limited_start, 1),
            ])
