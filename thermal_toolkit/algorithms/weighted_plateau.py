"""Global bin-weighted plateau mapping over the observed input DN range.

Engineering variant, not a reproduction of a published SAPHE/ADPHE formula.
Plateau factors scale with pixel count / populated levels of the original input.
Only populated bins receive a lower floor or redistributed counts.
Bright weighting uses the original-input percentile and interval midpoints.
Optional Gain Limit caps rounded LUT steps in exact tenths of Gray/DN,
anchors the original lower-median DN, and never stretches again.
Single-bin mode uses H'(i); adjacent-bin mode uses H'(i)+H'(i+1).
"""

import numpy as np


def weighted_plateau_u8(
    image, input_max,
    plateau_upper=1.0, plateau_lower=0.1,
    plateau_redistribute=False, interval_data=None, plateau_gain_limit=0.0,
    plateau_bright_weight=1.0, plateau_bright_percentile=80.0,
    plateau_weight_mode="adjacent",
):
    image = np.asarray(image)
    if image.ndim != 2 or not image.size:
        raise ValueError("Plateau mapping requires a non-empty 2D image.")
    if int(input_max) != input_max or not 1 <= input_max <= 65535:
        raise ValueError("Input maximum must be an integer from 1 to 65535.")
    input_max = int(input_max)
    upper, lower = float(plateau_upper), float(plateau_lower)
    bright_weight, bright_percentile = float(plateau_bright_weight), float(plateau_bright_percentile)
    if bright_weight not in (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
        raise ValueError("Bright Weight must be 1.05 or 1.0 to 2.0 in steps of 0.1.")
    if not 0 <= bright_percentile <= 100:
        raise ValueError("Bright Threshold Percentile must be between 0 and 100.")
    gain_limit = float(plateau_gain_limit)
    if (not np.isfinite(gain_limit) or not 0 <= gain_limit <= 4
            or not np.isclose(gain_limit * 10, round(gain_limit * 10), rtol=0, atol=1e-9)):
        raise ValueError("Gain Limit must be 0.0 (Off) to 4.0 Gray/DN in steps of 0.1.")
    gain_units = int(round(gain_limit * 10))
    gain_limit = gain_units / 10.0
    if plateau_weight_mode not in ("adjacent", "single"):
        raise ValueError("Weight mode must be adjacent or single.")
    if not np.isfinite(upper) or not 0.1 <= upper <= 100:
        raise ValueError("Upper Plateau must be 0.1..100 times the populated-bin average.")
    if not np.isfinite(lower) or not 0 <= lower <= 1:
        raise ValueError("Lower Plateau must be 0..1 times the populated-bin average.")
    if lower > upper:
        raise ValueError("Lower Plateau must not exceed Upper Plateau.")

    source = np.clip(image, 0, input_max).astype(np.int64)
    bright_threshold = float(np.percentile(source, bright_percentile))
    first, last = int(source.min()), int(source.max())
    original = np.bincount(source.ravel(), minlength=input_max + 1)
    histogram = original[first:last + 1]
    populated = histogram > 0
    average = float(source.size) / int(populated.sum())
    upper_count = max(1, int(np.floor(upper * average + 0.5)))
    lower_count = min(upper_count, int(np.floor(lower * average + 0.5)))
    processed = np.minimum(histogram, upper_count).astype(np.int64)
    excess = int((histogram - processed).sum())
    redistributed = 0
    if plateau_redistribute:
        remaining = min(excess, int((upper_count - processed[populated]).sum()))
        redistributed = remaining
        while remaining:
            eligible = np.flatnonzero(populated & (processed < upper_count))
            batch, residual = divmod(remaining, eligible.size)
            increments = np.minimum(upper_count - processed[eligible], batch)
            if residual:
                positions = np.arange(residual) * eligible.size // residual
                increments[positions] = np.minimum(increments[positions] + 1,
                                                   upper_count - processed[eligible[positions]])
            processed[eligible] += increments
            remaining -= int(increments.sum())
    floor_added = int(np.maximum(lower_count - processed[populated], 0).sum())
    processed[populated] = np.maximum(processed[populated], lower_count)
    weights = (processed[:-1] + processed[1:] if plateau_weight_mode == "adjacent"
               else processed[:-1].copy())
    before_bright_weights = weights
    if bright_weight != 1.0:
        weights = weights.astype(np.float64)
        midpoints = np.arange(first, last, dtype=np.float64) + 0.5
        weights[midpoints >= bright_threshold] *= bright_weight
    lut = np.zeros(input_max + 1, dtype=np.uint8)
    if first == last:
        # There is no scene contrast to stretch in a constant input.
        lut[:] = 128
    else:
        if bright_weight == 1.0:
            # Preserve the original integer round-half-up path exactly when Off.
            cumulative = np.concatenate(([0], np.cumsum(weights, dtype=np.int64)))
            total = int(cumulative[-1])
            lut[first:last + 1] = ((cumulative * 255 + total // 2) // total).astype(np.uint8)
        else:
            cumulative = np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64)))
            lut[first:last + 1] = np.clip(
                np.floor(cumulative * 255.0 / cumulative[-1] + 0.5), 0, 255,
            ).astype(np.uint8)
        lut[last + 1:] = 255
    before_gain_lut = lut
    limited_lut_float = lut.astype(np.float64) if interval_data is not None else None
    anchor = anchor_gray = None
    if gain_units:
        steps = np.minimum(np.diff(lut.astype(np.int64)) * 10, gain_units)
        limited = np.concatenate(([0], np.cumsum(steps)))
        anchor = int(np.searchsorted(np.cumsum(original), (source.size + 1) // 2))
        anchor_gray = int(lut[anchor])
        limited += anchor_gray * 10 - limited[anchor]
        limited = np.clip(limited, 0, 2550)
        if interval_data is not None:
            limited_lut_float = limited / 10.0
        lut = ((limited + 5) // 10).astype(np.uint8)
    mapped = np.ascontiguousarray(lut[source])
    info = dict(
        mapping_first_dn=first, mapping_last_dn=last,
        unique_levels=int(populated.sum()), histogram_average_count=average,
        upper_plateau_count=upper_count,
        weight_mode=plateau_weight_mode,
        weight_formula="H'(i)+H'(i+1)" if plateau_weight_mode == "adjacent" else "H'(i)",
        lower_plateau_count=lower_count, clipped_histogram_count=excess,
        redistributed_histogram_count=redistributed,
        discarded_histogram_count=excess - redistributed,
        floor_added_histogram_count=floor_added,
        processed_histogram_total=int(processed.sum()),
        histogram_peak_after_plateau=int(processed.max()),
        redistribution_scope="populated_bins" if plateau_redistribute else "none",
        grid=1, total_interval_weight=float(weights.sum()) if bright_weight != 1.0 else int(weights.sum()),
        before_bright_total_interval_weight=int(before_bright_weights.sum()),
        bright_weight=bright_weight, bright_percentile=bright_percentile,
        bright_threshold_dn=bright_threshold,
        output_min=int(mapped.min()), output_max=int(mapped.max()),
        constant_input_range=first == last,
        gain_limit=gain_limit, gain_limit_domain="before_output_rounding",
        gain_anchor_dn=anchor, gain_anchor_gray=anchor_gray,
        gain_limit_applied=bool(np.any(lut != before_gain_lut)),
    )
    if interval_data is not None:
        interval_data.clear()
        full_processed = np.zeros(input_max + 1, dtype=np.int64)
        full_processed[first:last + 1] = processed
        full_weights = np.zeros(input_max, dtype=weights.dtype)
        full_weights[first:last] = weights
        full_before_bright = np.zeros(input_max, dtype=np.int64)
        full_before_bright[first:last] = before_bright_weights
        interval_data.update(histogram=original, processed_histogram=full_processed,
                             weights=full_weights, lut=lut,
                             before_bright_weights=full_before_bright,
                             source_histogram=np.bincount(source.ravel(), minlength=input_max + 1),
                             before_gain_lut=before_gain_lut,
                             limited_lut_float=limited_lut_float)
    return mapped, info
