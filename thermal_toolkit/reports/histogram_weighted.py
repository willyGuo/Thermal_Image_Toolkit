"""reports / histogram_weighted."""

import csv
import numpy as np
from thermal_toolkit.algorithms.histogram import histogram_weighted_mapping_native_u8
from thermal_toolkit.algorithms.weighted_grid import weighted_tiles


def _interval_rows(image, input_max, parameters):
    """Export all native-DN intervals only when explicitly saving results.

    Original means no clipping or bright weighting. Before-bright includes
    clipping/redistribution. Ratios are normalized independently (sum = 1).
    Original intervals outside the populated range have zero allocation;
    redistributed intervals may have weights throughout the full input range.
    dn_start_percentile is 100 * count(input <= dn_start) / pixel_count,
    measured on the original input before clipping or weighting.
    Existing weight ratios/output spans remain before gain limiting. Appended
    LUT columns compare the rounded output before/after the final gain limit.
    limited_output_span_before_rounding retains fractional gain-limited spans.
    """
    data = {}
    histogram_weighted_mapping_native_u8(
        image, input_max,
        parameters["histogram_weighted_clip"],
        parameters["histogram_weighted_redistribute"],
        parameters["histogram_weighted_bright_weight"],
        parameters["histogram_weighted_bright_percentile"],
        interval_data=data,
        gain_limit=parameters.get("histogram_weighted_gain_limit", 0),
    )
    weights = [data[key] for key in (
        "original_weights", "before_bright_weights", "adjusted_weights"
    )]
    ratios = [w / w.sum() if w.sum() > 0 else np.zeros_like(w) for w in weights]
    source = np.clip(image, 0, input_max).astype(np.int64).reshape(-1)
    histogram = np.bincount(source, minlength=int(input_max) + 1)
    percentiles = (
        np.cumsum(histogram, dtype=np.int64) * (100.0 / source.size)
        if source.size else np.zeros(int(input_max) + 1, dtype=np.float64)
    )
    for i in range(int(input_max)):
        multiplier = data["bright_weight"] if i + 0.5 >= data["bright_threshold_dn"] else 1.0
        before_start, before_end = (int(v) for v in data["before_gain_lut"][i:i + 2])
        final_start, final_end = (int(v) for v in data["final_lut"][i:i + 2])
        yield [
            i, i + 1, float(percentiles[i]),
            data["bright_percentile"], data["bright_threshold_dn"],
            multiplier, *(float(w[i]) for w in weights),
            *(float(r[i]) for r in ratios),
            float(ratios[0][i] * 255), float(ratios[2][i] * 255),
            parameters.get("histogram_weighted_gain_limit", 0),
            before_start, before_end, before_end - before_start,
            final_start, final_end, final_end - final_start,
            round(float(data["limited_lut_float"][i + 1] - data["limited_lut_float"][i]), 1),
        ]


def save_histogram_weighted_intervals(path, image, input_max, parameters):
    """Save global intervals or per-tile intervals before spatial interpolation."""
    grid = parameters.get("histogram_weighted_grid", 1)
    with open(path, "w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.writer(destination)
        prefix = ["tile_y", "tile_x"] if grid > 1 else []
        writer.writerow(prefix + [
            "dn_start", "dn_end", "dn_start_percentile", "bright_percentile", "bright_threshold_dn",
            "bright_multiplier", "original_weight", "before_bright_weight",
            "adjusted_weight", "original_ratio", "before_bright_ratio",
            "adjusted_ratio", "original_output_span", "adjusted_output_span",
            "gain_limit_gray_per_dn", "before_gain_output_start", "before_gain_output_end",
            "before_gain_output_span", "final_output_start", "final_output_end", "final_output_span",
            "limited_output_span_before_rounding",
        ])
        for ty, tx, tile in weighted_tiles(image, grid):
            for row in _interval_rows(tile, input_max, parameters):
                writer.writerow(([ty, tx] if grid > 1 else []) + row)
