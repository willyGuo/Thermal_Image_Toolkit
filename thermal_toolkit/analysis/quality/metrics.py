"""analysis / quality / metrics."""

import numpy as np
from thermal_toolkit.core.opencv import cv2


def _robust_sigma(values):
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return 0.0
    median = float(np.median(samples))
    return float(1.4826 * np.median(np.abs(samples - median)))


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or weights.size != values.size:
        return 0.0
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float(np.median(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 0.0
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values[min(index, values.size - 1)])


def _quality_input_to_normalized(image, input_max):
    if input_max <= 0:
        raise ValueError("Quality analysis input_max must be positive.")
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("Quality analysis requires a non-empty gray image.")
    return np.clip(source.astype(np.float32), 0.0, float(input_max)) / float(input_max)


def _entropy_bits_from_normalized(image_f):
    gray_u8 = np.rint(np.clip(image_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    histogram = np.bincount(gray_u8.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray_u8.size
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _line_profile_seam_metrics(profile):
    """Measure repeated near-periodic discontinuity lines in one orientation."""
    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    if profile.size == 0:
        return {
            "ratio": 1.0,
            "strongest": 0.0,
            "baseline": 0.0,
            "peak_percent": 0.0,
            "peak_group_count": 0,
            "periodicity_score": 0.0,
            "penalty": 0.0,
        }

    baseline = float(np.median(profile))
    spread = _robust_sigma(profile)
    top_count = max(1, int(np.ceil(profile.size * 0.05)))
    strongest = np.partition(profile, profile.size - top_count)[-top_count:]
    strongest_mean = float(np.mean(strongest))
    denominator = max(baseline + spread, 1.0e-6)
    ratio = strongest_mean / denominator

    # Path A: very strong repeated line outliers.
    peak_threshold = baseline + 3.0 * max(spread, 1.0 / 255.0)
    peak_indices = np.flatnonzero(profile > peak_threshold)
    peak_percent = float(peak_indices.size * 100.0 / profile.size)
    peak_groups = []
    if peak_indices.size:
        start = previous = int(peak_indices[0])
        for index in peak_indices[1:]:
            index = int(index)
            if index <= previous + 1:
                previous = index
            else:
                peak_groups.append((start, previous))
                start = previous = index
        peak_groups.append((start, previous))

    peak_periodicity = 0.0
    if len(peak_groups) >= 4:
        centers = np.asarray(
            [(start + end) * 0.5 for start, end in peak_groups],
            dtype=np.float64,
        )
        gaps = np.diff(centers)
        mean_gap = float(np.mean(gaps)) if gaps.size else 0.0
        if mean_gap > 0.0:
            gap_cv = float(np.std(gaps) / mean_gap)
            regularity = float(np.exp(-((gap_cv / 0.30) ** 2)))
            repetition_support = float(
                np.clip((len(peak_groups) - 3) / 4.0, 0.0, 1.0)
            )
            peak_periodicity = 100.0 * regularity * repetition_support
    peak_strength_gate = float(np.clip((ratio - 1.10) / 1.80, 0.0, 1.0))
    peak_penalty = peak_periodicity * peak_strength_gate

    # Path B: a weaker but repeated comb pattern. This catches visible grid/block lines
    # that are individually weaker than natural object boundaries. The P25 sequence
    # strength requirement prevents a few accidental peaks from creating a false grid.
    smooth = cv2.GaussianBlur(
        profile.reshape(1, -1),
        (0, 0),
        1.0,
    ).reshape(-1)
    smooth_baseline = float(np.median(smooth))
    smooth_spread = _robust_sigma(smooth)
    residual = np.maximum(smooth - smooth_baseline, 0.0)
    positive = residual[residual > 0.0]
    comb_penalty = 0.0
    comb_periodicity = 0.0
    if positive.size >= 4:
        cap = float(np.percentile(positive, 85.0))
        if cap > 1.0e-10:
            normalized_residual = np.minimum(residual, cap) / cap
            sample_count = smooth.size
            comb_denominator = max(smooth_baseline + smooth_spread, 1.0e-6)
            # Search a compact set of plausible grid densities (4..20 cells) and
            # evaluate all phases at once. This is much faster than scanning every
            # possible period/phase pair in Python.
            tested_periods = set()
            for cell_count in range(4, 21):
                period = int(round(sample_count / float(cell_count)))
                period = max(4, period)
                if period in tested_periods:
                    continue
                tested_periods.add(period)
                repetitions = sample_count // period
                if repetitions < 4:
                    continue
                usable = repetitions * period
                normalized_matrix = normalized_residual[:usable].reshape(
                    repetitions, period
                )
                actual_matrix = smooth[:usable].reshape(repetitions, period)
                repetition_support = float(
                    np.clip((repetitions - 3) / 5.0, 0.0, 1.0)
                )
                pattern_support = (
                    0.60 * np.mean(normalized_matrix, axis=0)
                    + 0.40 * np.percentile(normalized_matrix, 25.0, axis=0)
                ) * repetition_support
                sequence_strength_ratio = (
                    np.percentile(actual_matrix, 25.0, axis=0)
                    / comb_denominator
                )
                strength_gate = np.clip(
                    (sequence_strength_ratio - 1.50) / 4.0,
                    0.0,
                    1.0,
                )
                candidate_penalty = 100.0 * pattern_support * strength_gate
                best_phase = int(np.argmax(candidate_penalty))
                best_penalty = float(candidate_penalty[best_phase])
                if best_penalty > comb_penalty:
                    comb_penalty = best_penalty
                    comb_periodicity = float(
                        100.0 * pattern_support[best_phase]
                    )

    periodicity_score = max(peak_periodicity, comb_periodicity)
    penalty = float(np.clip(max(peak_penalty, comb_penalty), 0.0, 100.0))
    return {
        "ratio": float(ratio),
        "strongest": strongest_mean,
        "baseline": baseline,
        "peak_percent": peak_percent,
        "peak_group_count": int(len(peak_groups)),
        "periodicity_score": float(periodicity_score),
        "penalty": penalty,
    }


def _whole_image_seam_ratio(image):
    """Detect repeated horizontal/vertical block seams while tolerating natural edges."""
    image_f = np.asarray(image, dtype=np.float32)
    candidates = []
    if image_f.shape[1] > 1:
        dx = np.abs(np.diff(image_f, axis=1))
        candidates.append(_line_profile_seam_metrics(np.mean(dx, axis=0)))
    if image_f.shape[0] > 1:
        dy = np.abs(np.diff(image_f, axis=0))
        candidates.append(_line_profile_seam_metrics(np.mean(dy, axis=1)))
    if not candidates:
        return 1.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0

    # Report the orientation with the strongest artifact evidence. If none is periodic,
    # use the strongest ratio only for diagnostics, not for the penalty.
    selected = max(candidates, key=lambda item: (item["penalty"], item["ratio"]))
    max_periodicity = max(item["periodicity_score"] for item in candidates)
    max_group_count = max(item["peak_group_count"] for item in candidates)
    max_penalty = max(item["penalty"] for item in candidates)
    return (
        float(selected["ratio"]),
        float(selected["strongest"]),
        float(selected["baseline"]),
        float(selected["peak_percent"]),
        float(max_periodicity),
        int(max_group_count),
        float(max_penalty),
    )


def _edge_halo_overshoot_ratio(
    source_f,
    gradient_x,
    gradient_y,
    edge_mask,
    max_samples=4096,
):
    """Estimate true edge halo/ringing by sampling across the edge normal.

    A normal edge should transition between its two plateaus without leaving their range.
    Sharpening halos create overshoot/undershoot outside that local plateau range.
    """
    source_f = np.asarray(source_f, dtype=np.float32)
    edge_mask = np.asarray(edge_mask, dtype=bool)
    ys, xs = np.nonzero(edge_mask)
    if ys.size == 0:
        return 0.0, 0

    gx = gradient_x[ys, xs].astype(np.float32)
    gy = gradient_y[ys, xs].astype(np.float32)
    magnitude = np.sqrt(gx * gx + gy * gy)
    valid_direction = magnitude > 1e-6
    ys = ys[valid_direction]
    xs = xs[valid_direction]
    gx = gx[valid_direction]
    gy = gy[valid_direction]
    magnitude = magnitude[valid_direction]
    if ys.size == 0:
        return 0.0, 0

    if ys.size > int(max_samples):
        keep = np.argpartition(magnitude, -int(max_samples))[-int(max_samples):]
        ys = ys[keep]
        xs = xs[keep]
        gx = gx[keep]
        gy = gy[keep]
        magnitude = magnitude[keep]

    nx = gx / np.maximum(magnitude, 1e-6)
    ny = gy / np.maximum(magnitude, 1e-6)
    profiles = []
    for distance in (-4, -3, -2, -1, 0, 1, 2, 3, 4):
        map_x = (xs.astype(np.float32) + nx * distance).reshape(-1, 1)
        map_y = (ys.astype(np.float32) + ny * distance).reshape(-1, 1)
        samples = cv2.remap(
            source_f,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).reshape(-1)
        profiles.append(samples)
    profile = np.stack(profiles, axis=1)

    left_plateau = np.mean(profile[:, 0:2], axis=1)
    right_plateau = np.mean(profile[:, -2:], axis=1)
    edge_contrast = np.abs(right_plateau - left_plateau)
    valid = edge_contrast >= 0.03
    if np.count_nonzero(valid) < 16:
        return 0.0, int(np.count_nonzero(valid))

    center = profile[valid, 2:7]
    left_plateau = left_plateau[valid]
    right_plateau = right_plateau[valid]
    edge_contrast = edge_contrast[valid]
    lower = np.minimum(left_plateau, right_plateau)
    upper = np.maximum(left_plateau, right_plateau)
    overshoot_high = np.maximum(np.max(center, axis=1) - upper, 0.0)
    overshoot_low = np.maximum(lower - np.min(center, axis=1), 0.0)
    severity = np.maximum(overshoot_high, overshoot_low) / np.maximum(
        edge_contrast, 0.03
    )

    # A P75/P90 blend is robust to a few textured edges while still reacting when
    # overshoot is repeated across many real edges.
    severity_p75 = float(np.percentile(severity, 75.0))
    severity_p90 = float(np.percentile(severity, 90.0))
    ratio = 0.70 * severity_p75 + 0.30 * severity_p90
    return float(ratio), int(severity.size)


def _isolated_impulse_analysis(image_f, max_component_area=4):
    """Detect isolated hot/cold impulse pixels without treating edge halos as impulses."""
    image_f = np.asarray(image_f, dtype=np.float32)
    median = cv2.medianBlur(image_f, 3)
    residual = image_f - median
    sigma = _robust_sigma(residual)
    threshold = max(6.0 * sigma, 3.0 / 255.0)
    candidate = (np.abs(residual) > threshold).astype(np.uint8)
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )
    isolated_pixels = 0
    isolated_components = 0
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area <= int(max_component_area):
            isolated_pixels += area
            isolated_components += 1
    isolated_percent = float(
        isolated_pixels * 100.0 / image_f.size
    )
    score = float(100.0 * np.exp(-isolated_percent / 0.25))
    return {
        "score": score,
        "percent": isolated_percent,
        "pixel_count": int(isolated_pixels),
        "component_count": int(isolated_components),
        "threshold": float(threshold),
        "sigma": float(sigma),
    }


def _row_column_banding_sigma(residual, flat_mask):
    residual = np.asarray(residual, dtype=np.float32)
    flat_mask = np.asarray(flat_mask, dtype=bool)
    values = []
    minimum_row_pixels = max(8, residual.shape[1] // 12)
    minimum_col_pixels = max(8, residual.shape[0] // 12)
    for row in range(residual.shape[0]):
        mask = flat_mask[row]
        if np.count_nonzero(mask) >= minimum_row_pixels:
            values.append(float(np.median(residual[row][mask])))
    for column in range(residual.shape[1]):
        mask = flat_mask[:, column]
        if np.count_nonzero(mask) >= minimum_col_pixels:
            values.append(float(np.median(residual[:, column][mask])))
    return _robust_sigma(values)


def _full_flat_region_analysis(base, detail, uniform_residual, flat_mask, gray_bands=8):
    """Split flat pixels by gray level and connected region before measuring noise.

    This prevents large flat areas at different brightness levels (e.g. sky and wall) from
    being mixed into one noise population. It also keeps spatially separate regions apart.
    """
    flat_values = base[flat_mask]
    if flat_values.size == 0:
        return [], 0
    low, high = np.percentile(flat_values, [1.0, 99.0])
    if high <= low:
        low = float(np.min(flat_values))
        high = float(np.max(flat_values))
    if high <= low:
        low, high = 0.0, 1.0

    minimum_area = max(96, int(np.ceil(base.size * 0.0015)))
    regions = []
    for band_index in range(gray_bands):
        band_low = low + (high - low) * band_index / gray_bands
        band_high = low + (high - low) * (band_index + 1) / gray_bands
        if band_index == gray_bands - 1:
            band_mask = flat_mask & (base >= band_low) & (base <= band_high)
        else:
            band_mask = flat_mask & (base >= band_low) & (base < band_high)
        if np.count_nonzero(band_mask) < minimum_area:
            continue
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            band_mask.astype(np.uint8), connectivity=8
        )
        for label_id in range(1, count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            local_labels = labels[y:y + height, x:x + width]
            local_mask = local_labels == label_id
            local_detail = detail[y:y + height, x:x + width][local_mask]
            local_uniform = uniform_residual[y:y + height, x:x + width][local_mask]
            local_base = base[y:y + height, x:x + width][local_mask]
            regions.append(
                {
                    "band": int(band_index),
                    "area": area,
                    "mean_level": float(np.mean(local_base)),
                    "noise_sigma": _robust_sigma(local_detail),
                    "uniform_sigma": _robust_sigma(local_uniform),
                }
            )

    if not regions and np.count_nonzero(flat_mask) >= 16:
        regions.append(
            {
                "band": 0,
                "area": int(np.count_nonzero(flat_mask)),
                "mean_level": float(np.mean(base[flat_mask])),
                "noise_sigma": _robust_sigma(detail[flat_mask]),
                "uniform_sigma": _robust_sigma(uniform_residual[flat_mask]),
            }
        )

    regions.sort(key=lambda item: item["area"], reverse=True)
    regions = regions[:48]
    return regions, len({item["band"] for item in regions})
