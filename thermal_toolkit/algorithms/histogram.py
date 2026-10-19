"""algorithms / histogram."""

import numpy as np
from thermal_toolkit.algorithms.fieldscale import FIELDSCALE_RAW14_MAX
from thermal_toolkit.algorithms.weighted_aci import weighted_aci_clip


def histogram_equalization_native_u8(image, input_max, exclude_zero=False):
    source = np.clip(image, 0, input_max).astype(np.int64)
    valid = source != 0 if exclude_zero else np.ones(source.shape, dtype=bool)
    values = source[valid]
    if values.size == 0:
        return np.zeros(source.shape, dtype=np.uint8)
    histogram = np.bincount(values.reshape(-1), minlength=int(input_max) + 1)
    cdf = np.cumsum(histogram, dtype=np.int64)
    populated = np.flatnonzero(histogram)
    if populated.size <= 1:
        result = np.zeros(source.shape, dtype=np.uint8)
    else:
        cdf_min = int(cdf[populated[0]])
        denominator = int(values.size) - cdf_min
        if denominator <= 0:
            result = np.zeros(source.shape, dtype=np.uint8)
        else:
            lut = np.clip(
                np.rint((cdf - cdf_min) * 255.0 / denominator), 0, 255
            ).astype(np.uint8)
            result = lut[source]
    if exclude_zero:
        result[~valid] = 0
    return result


def histogram_weighted_mapping_native_u8(
    image, input_max, clip_limit=2.0, redistribute=True, bright_weight=1.0,
    bright_percentile=80.0, interval_data=None, lut_data=None, gain_limit=0,
):
    """Map native DN to 8-bit using adjacent-bin histogram weights.

    For adjacent native-DN bins i and i+1:
        W(i) = H(i) + H(i+1)
    Each interval receives a fraction W(i) / sum(W) of the 0..255 output
    range. Without redistribution the first/last populated DN map to 0/255;
    with redistribution, input DN 0/input_max map to 0/255.
    When Clip Limit is positive, the histogram is clipped
    and, when redistribute is enabled, its excess is redistributed across the
    full input DN range, including empty bins, before weights are calculated.
    Intervals whose midpoint is >= the selected input percentile receive bright_weight;
    the modified interval weights are then normalized to the full output range.
    Optional gain_limit (0.0..4.0 Gray/DN in tenths; 0 disables) caps adjacent steps of
    the rounded LUT, preserving the original lower-median DN's output brightness.
    The limited LUT is clipped to 0..255 without rescaling. Outside the mapping
    range it holds the endpoint values. Empty inputs keep their zero output;
    constant inputs use full-range mapping when automatic redistribution is active.
    Fractional steps accumulate in exact tenths, then round half up to uint8.
    The fractional bound applies before output rounding, not to integer pixel steps.
    """
    source = np.clip(image, 0, input_max).astype(np.int64)
    gain_limit = float(gain_limit)
    if (not np.isfinite(gain_limit) or not 0.0 <= gain_limit <= 4.0
            or not np.isclose(gain_limit * 10, round(gain_limit * 10), rtol=0, atol=1e-9)):
        raise ValueError("Histogram-weighted Gain Limit must be 0.0 (Off) to 4.0 Gray/DN in steps of 0.1.")
    gain_units = int(round(gain_limit * 10))
    gain_limit = gain_units / 10.0
    gain_info = {
        "gain_limit": gain_limit,
        "gain_limit_domain": "before_output_rounding",
        "gain_anchor_dn": None,
        "gain_anchor_gray": None,
        "first_output_gray": 0,
        "last_output_gray": 0,
    }
    if lut_data is not None:
        lut_data.clear()
        lut_data["lut"] = np.zeros(int(input_max) + 1, dtype=np.uint8)
    bright_weight = float(bright_weight)
    if bright_weight not in (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0):
        raise ValueError("Bright weight must be 1.05 or 1.0 to 2.0 in steps of 0.1.")
    bright_percentile = float(bright_percentile)
    if not 0.0 <= bright_percentile <= 100.0:
        raise ValueError("Bright percentile must be between 0 and 100.")
    bright_threshold = float(np.percentile(source, bright_percentile)) if source.size else 0.0
    bright_info = {"bright_percentile": bright_percentile, "bright_threshold_dn": bright_threshold,
                   "bright_weight": bright_weight}
    clip_limit = float(clip_limit)
    if clip_limit < 0.0:
        raise ValueError("Histogram-weighted Clip Limit must be non-negative.")
    histogram = np.bincount(source.reshape(-1), minlength=int(input_max) + 1).astype(np.int64)
    populated = np.flatnonzero(histogram)
    histogram_peak_count = int(histogram.max()) if histogram.size else 0
    aci_info = weighted_aci_clip(source, int(populated.size))
    if aci_info is not None:
        gain_info.update(aci_info)
    histogram_average_count = (
        float(source.size) / populated.size if populated.size else 0.0
    )
    peak_ratio = (
        histogram_peak_count / histogram_average_count
        if histogram_average_count > 0.0
        else 0.0
    )
    if interval_data is not None:
        interval_data.clear()
        interval_data.update(
            original_weights=np.zeros(int(input_max), dtype=np.float64),
            before_bright_weights=np.zeros(int(input_max), dtype=np.float64),
            adjusted_weights=np.zeros(int(input_max), dtype=np.float64),
            before_gain_lut=np.zeros(int(input_max) + 1, dtype=np.uint8),
            final_lut=np.zeros(int(input_max) + 1, dtype=np.uint8),
            limited_lut_float=np.zeros(int(input_max) + 1, dtype=np.float64),
            **bright_info,
        )
    if populated.size <= 1 and not (aci_info is not None and redistribute):
        return np.zeros(source.shape, dtype=np.uint8), {
            "first_populated_dn": int(populated[0]) if populated.size else 0,
            "last_populated_dn": int(populated[0]) if populated.size else 0,
            **gain_info,
            **bright_info,
            "total_interval_weight": 0,
            "clip_limit": clip_limit,
            "clip_count_per_bin": None,
            "unique_levels": int(populated.size),
            "histogram_peak_count": histogram_peak_count,
            "histogram_average_count": histogram_average_count,
            "peak_ratio": peak_ratio,
        }

    first = int(populated[0])
    last = int(populated[-1])
    weighted_histogram = histogram[first:last + 1]
    mapping_first, mapping_last = first, last
    clip_count = None
    if clip_limit > 0.0 or aci_info is not None:
        clip_count = max(
            1,
            int(round(aci_info['aci_clip_count'] if aci_info is not None
                      else clip_limit * source.size / populated.size)),
        )
        if redistribute:
            weighted_histogram = _redistribute_histogram_with_cap(histogram, clip_count)
            mapping_first, mapping_last = 0, int(input_max)
        else:
            weighted_histogram = np.minimum(weighted_histogram, clip_count)
    weights = weighted_histogram[:-1] + weighted_histogram[1:]
    if interval_data is not None:
        interval_data["original_weights"][first:last] = (
            histogram[first:last] + histogram[first + 1:last + 1]
        )
        interval_data["before_bright_weights"][mapping_first:mapping_last] = weights
    if bright_weight != 1.0:
        weights = weights.astype(np.float64)
        interval_midpoints = np.arange(mapping_first, mapping_last, dtype=np.float64) + 0.5
        weights[interval_midpoints >= bright_threshold] *= bright_weight
    if interval_data is not None:
        interval_data["adjusted_weights"][mapping_first:mapping_last] = weights
    total_weight = float(weights.sum()) if bright_weight != 1.0 else int(weights.sum())
    if total_weight <= 0:
        return np.zeros(source.shape, dtype=np.uint8), {
            "first_populated_dn": first,
            "last_populated_dn": last,
            **gain_info,
            **bright_info,
            "total_interval_weight": 0,
            "clip_limit": clip_limit,
            "clip_count_per_bin": clip_count,
            "unique_levels": int(populated.size),
            "histogram_peak_count": histogram_peak_count,
            "histogram_average_count": histogram_average_count,
            "peak_ratio": peak_ratio,
        }

    lut = np.zeros(int(input_max) + 1, dtype=np.uint8)
    cumulative = np.concatenate(([0], np.cumsum(weights, dtype=np.float64 if bright_weight != 1.0 else np.int64)))
    mapped_range = np.rint(cumulative * 255.0 / total_weight)
    lut[mapping_first:mapping_last + 1] = np.clip(mapped_range, 0, 255).astype(np.uint8)
    lut[mapping_last + 1:] = 255
    if interval_data is not None:
        interval_data["before_gain_lut"] = lut
        interval_data["limited_lut_float"] = lut.astype(np.float64)
    if gain_limit:
        # Preserve the existing rounded mapping; accumulate fractional caps exactly
        # in tenths. Integer gain settings remain bit-exact with the former limiter.
        steps = np.minimum(np.diff(lut.astype(np.int64)) * 10, gain_units)
        limited = np.concatenate(([0], np.cumsum(steps)))
        # Use an actually populated DN, including for an even number of pixels.
        anchor = int(np.searchsorted(np.cumsum(histogram), (source.size + 1) // 2))
        anchor_gray = int(lut[anchor])
        limited += anchor_gray * 10 - limited[anchor]
        limited = np.clip(limited, 0, 2550)
        if interval_data is not None:
            interval_data["limited_lut_float"] = limited / 10.0
        lut = ((limited + 5) // 10).astype(np.uint8)
        gain_info.update(gain_anchor_dn=anchor, gain_anchor_gray=anchor_gray)
    gain_info.update(first_output_gray=int(lut[first]), last_output_gray=int(lut[last]))
    if interval_data is not None:
        interval_data["final_lut"] = lut
    if lut_data is not None:
        lut_data["lut"] = lut
    return lut[source], {
        "first_populated_dn": first,
        "last_populated_dn": last,
        **gain_info,
        **bright_info,
        "total_interval_weight": total_weight,
        "clip_limit": clip_limit,
        "clip_count_per_bin": clip_count,
        "unique_levels": int(populated.size),
        "histogram_peak_count": histogram_peak_count,
        "histogram_average_count": histogram_average_count,
        "peak_ratio": peak_ratio,
        "histogram_peak_after_clip": int(weighted_histogram.max()),
        "redistribution_scope": "full_input_range" if clip_count is not None and redistribute else "none",
        "mapping_first_dn": mapping_first,
        "mapping_last_dn": mapping_last,
        "discarded_histogram_count": int(source.size - weighted_histogram.sum()),
    }


def _redistribute_histogram_with_cap(histogram, clip_count):
    """Fill all DN bins with spare capacity, preserving the clipping cap."""
    result = np.minimum(histogram, clip_count).astype(np.int64)
    excess = int(np.sum(histogram) - np.sum(result))
    excess = min(excess, int(np.sum(clip_count - result)))
    while excess:
        available = np.flatnonzero(result < clip_count)
        batch, remainder = divmod(excess, available.size)
        increments = np.minimum(clip_count - result[available], batch)
        if remainder:
            # Spread residual counts evenly over bins with spare capacity.
            indices = np.arange(remainder) * available.size // remainder
            increments[indices] = np.minimum(increments[indices] + 1, clip_count - result[available[indices]])
        result[available] += increments
        excess -= int(increments.sum())
    return result


def _clahe_clip_histogram(histogram, clip_count):
    """Clip a CLAHE histogram and distribute its excess uniformly."""
    histogram = histogram.astype(np.int64, copy=True)
    excess = np.maximum(histogram - clip_count, 0).sum()
    np.minimum(histogram, clip_count, out=histogram)
    if excess == 0:
        return histogram
    batch, excess = divmod(int(excess), histogram.size)
    if batch:
        histogram += batch
    if excess:
        step = max(histogram.size // excess, 1)
        histogram[np.arange(0, histogram.size, step)[:excess]] += 1
    return histogram


def clahe_unique_levels_to_8bit(
    image, input_max, tiles_x=8, tiles_y=8, clip_limit=2.0
):
    """Apply CLAHE with a per-tile Pixel Count / Unique Levels clip count."""
    if image.ndim != 2:
        raise ValueError("CLAHE input must be a 2D image.")
    if tiles_x < 1 or tiles_y < 1 or clip_limit < 0:
        raise ValueError(
            "CLAHE requires positive grid dimensions and a non-negative clip limit."
        )
    input_max = max(1, int(input_max))
    source_bits = max(1, min(16, input_max.bit_length()))
    source = np.clip(image, 0, input_max)
    if input_max == FIELDSCALE_RAW14_MAX:
        histogram_bins = FIELDSCALE_RAW14_MAX + 1
        bin_source = source.astype(np.uint16)
        implementation = "direct_raw14_unique_levels"
    else:
        histogram_bins = 256
        bin_source = np.rint(
            source.astype(np.float64) * 255.0 / input_max
        ).astype(np.uint8)
        implementation = "direct_256bin_unique_levels"
    height, width = source.shape
    tile_width = (width + tiles_x - 1) // tiles_x
    tile_height = (height + tiles_y - 1) // tiles_y
    padded_width = tile_width * tiles_x
    padded_height = tile_height * tiles_y
    image = np.pad(
        bin_source,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    tile_pixels = tile_width * tile_height
    luts = np.empty((tiles_y, tiles_x, histogram_bins), dtype=np.uint8)
    tile_unique_levels = []
    tile_clip_counts = []

    for tile_y in range(tiles_y):
        y0 = tile_y * tile_height
        y1 = y0 + tile_height
        for tile_x in range(tiles_x):
            x0 = tile_x * tile_width
            x1 = x0 + tile_width
            histogram = np.bincount(
                image[y0:y1, x0:x1].ravel(), minlength=histogram_bins
            )
            unique_levels = max(1, int(np.count_nonzero(histogram)))
            clip_count = (
                max(
                    1,
                    int(round(clip_limit * tile_pixels / unique_levels)),
                )
                if clip_limit > 0
                else 0
            )
            tile_unique_levels.append(unique_levels)
            tile_clip_counts.append(clip_count)
            clipped_histogram = (
                _clahe_clip_histogram(histogram, clip_count)
                if clip_count > 0
                else histogram
            )
            cdf = np.cumsum(clipped_histogram, dtype=np.int64)
            luts[tile_y, tile_x] = np.clip(
                np.rint(cdf * 255.0 / tile_pixels), 0, 255
            ).astype(np.uint8)

    def interpolation_indices(length, tile_size, tile_count):
        grid = (np.arange(length, dtype=np.float32) + 0.5) / tile_size - 0.5
        lower = np.floor(grid).astype(np.int32)
        weight = grid - lower
        upper = lower + 1
        before = lower < 0
        lower[before] = upper[before] = 0
        weight[before] = 0.0
        after = upper >= tile_count
        lower[after] = upper[after] = tile_count - 1
        weight[after] = 0.0
        return lower, upper, weight

    x0, x1, weight_x = interpolation_indices(
        padded_width, tile_width, tiles_x
    )
    y0, y1, weight_y = interpolation_indices(
        padded_height, tile_height, tiles_y
    )
    output = np.empty((padded_height, padded_width), dtype=np.uint8)
    for y in range(padded_height):
        row = image[y]
        upper_left = luts[y0[y], x0, row].astype(np.float32)
        upper_right = luts[y0[y], x1, row].astype(np.float32)
        lower_left = luts[y1[y], x0, row].astype(np.float32)
        lower_right = luts[y1[y], x1, row].astype(np.float32)
        upper = upper_left + (upper_right - upper_left) * weight_x
        lower = lower_left + (lower_right - lower_left) * weight_x
        output[y] = np.clip(
            np.rint(upper + (lower - upper) * weight_y[y]), 0, 255
        ).astype(np.uint8)
    return output[:height, :width], {
        "clahe_implementation": implementation,
        "source_bits": source_bits,
        "histogram_bins": histogram_bins,
        "input_dn_per_histogram_bin": max(
            1, ((input_max + 1) + histogram_bins - 1) // histogram_bins
        ),
        "tile_pixels": tile_pixels,
        "tile_unique_levels_min": min(tile_unique_levels),
        "tile_unique_levels_max": max(tile_unique_levels),
        "tile_unique_levels_mean": float(np.mean(tile_unique_levels)),
        "internal_clip_count_min": min(tile_clip_counts),
        "internal_clip_count_max": max(tile_clip_counts),
        "internal_clip_count_mean": float(np.mean(tile_clip_counts)),
    }


def clahe_raw14_to_8bit(raw14, tiles_x=8, tiles_y=8, clip_limit=2.0):
    """Compatibility wrapper for direct RAW14 CLAHE."""
    mapped, info = clahe_unique_levels_to_8bit(
        raw14, FIELDSCALE_RAW14_MAX, tiles_x, tiles_y, clip_limit
    )
    return mapped, info["internal_clip_count_max"]
