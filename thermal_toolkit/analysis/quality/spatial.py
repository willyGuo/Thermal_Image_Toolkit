"""analysis / quality / spatial."""

import numpy as np
from thermal_toolkit.analysis.quality.metrics import (
    _edge_halo_overshoot_ratio,
    _entropy_bits_from_normalized,
    _full_flat_region_analysis,
    _isolated_impulse_analysis,
    _quality_input_to_normalized,
    _robust_sigma,
    _row_column_banding_sigma,
    _weighted_quantile,
    _whole_image_seam_ratio,
)
from thermal_toolkit.core.constants import ACTIVE_SPATIAL_QUALITY_WEIGHTS
from thermal_toolkit.core.opencv import cv2


def _spatial_overall_score(detail_score, contrast_score, noise_score, uniformity_score, clipping_score, artifact_score):
    scores = {
        "detail_clarity_score": detail_score,
        "contrast_quality_score": contrast_score,
        "flat_cleanliness_score": noise_score,
        "thermal_uniformity_score": uniformity_score,
        "clipping_control_score": clipping_score,
        "artifact_control_score": artifact_score,
    }
    return float(
        np.clip(
            sum(
                ACTIVE_SPATIAL_QUALITY_WEIGHTS[key] * scores[key]
                for key in scores
            )
            / 100.0,
            0.0,
            100.0,
        )
    )


def thermal_raw_quality_analysis(image, input_max):
    """Full no-reference spatial analysis for RAW/still-image comparison.

    The full profile keeps native resolution, separates base/detail layers, segments flat
    areas by both brightness and connectivity, mixes typical and worst-region statistics,
    and adds row/column banding diagnostics. Scores are relative same-scene indicators.
    """
    source_f = _quality_input_to_normalized(image, input_max)
    height, width = source_f.shape

    # Edge-preserving base/detail decomposition. Working in normalized 0..1 makes the
    # thresholds independent of RAW bit depth.
    base = cv2.bilateralFilter(source_f, 7, 0.08, 5.0)
    detail = source_f - base
    detail_abs = np.abs(detail)

    gradient_x = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y) / 4.0
    local_mean = cv2.blur(base, (9, 9))
    local_mean_square = cv2.blur(base * base, (9, 9))
    local_std = np.sqrt(np.maximum(local_mean_square - local_mean * local_mean, 0.0))

    edge_threshold = max(float(np.percentile(gradient, 75.0)), 0.015)
    edge_mask = gradient >= edge_threshold
    if np.count_nonzero(edge_mask) < max(16, source_f.size // 500):
        edge_mask = gradient > 0.005

    flat_gradient_threshold = float(
        np.clip(np.percentile(gradient, 35.0), 0.004, 0.020)
    )
    flat_std_threshold = float(
        np.clip(np.percentile(local_std, 50.0), 0.008, 0.035)
    )
    flat_mask = (
        (gradient <= flat_gradient_threshold)
        & (local_std <= flat_std_threshold)
    )
    # Keep high-gradient edge neighborhoods out of flat-noise statistics.
    edge_guard = cv2.dilate(edge_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    guarded_flat_mask = flat_mask & ~edge_guard
    if np.count_nonzero(guarded_flat_mask) >= max(32, source_f.size // 200):
        flat_mask = guarded_flat_mask
    elif np.count_nonzero(flat_mask) < max(16, source_f.size // 100):
        flat_mask = gradient <= flat_gradient_threshold

    broad_base = cv2.GaussianBlur(base, (0, 0), 5.0)
    uniform_residual = base - broad_base
    flat_regions, flat_gray_band_count = _full_flat_region_analysis(
        base, detail, uniform_residual, flat_mask, gray_bands=8
    )

    if flat_regions:
        region_areas = np.asarray([region["area"] for region in flat_regions], dtype=np.float64)
        region_noise = np.asarray([region["noise_sigma"] for region in flat_regions], dtype=np.float64)
        region_uniformity = np.asarray([region["uniform_sigma"] for region in flat_regions], dtype=np.float64)
        flat_noise_typical = _weighted_quantile(region_noise, region_areas, 0.50)
        # Worst-region terms are intentionally not area-weighted. Otherwise a large clean
        # sky can hide a smaller noisy wall/object. Very small fragments were already
        # rejected by the connected-component minimum-area rule above.
        flat_noise_worst = float(np.percentile(region_noise, 80.0))
        uniform_typical = _weighted_quantile(region_uniformity, region_areas, 0.50)
        uniform_worst = float(np.percentile(region_uniformity, 80.0))
        analyzed_flat_pixels = int(np.sum(region_areas))
    else:
        flat_noise_typical = _robust_sigma(detail[flat_mask]) if np.any(flat_mask) else 0.0
        flat_noise_worst = flat_noise_typical
        uniform_typical = _robust_sigma(uniform_residual[flat_mask]) if np.any(flat_mask) else 0.0
        uniform_worst = uniform_typical
        analyzed_flat_pixels = int(np.count_nonzero(flat_mask))

    noise_typical_score = 100.0 * np.exp(-flat_noise_typical / 0.025)
    noise_worst_score = 100.0 * np.exp(-flat_noise_worst / 0.025)
    robust_noise_score = float(
        np.clip(
            0.65 * noise_typical_score + 0.35 * noise_worst_score,
            0.0,
            100.0,
        )
    )
    impulse_result = _isolated_impulse_analysis(
        source_f,
        max_component_area=4,
    )
    impulse_outlier_score = float(impulse_result["score"])
    impulse_outlier_percent = float(impulse_result["percent"])
    impulse_threshold = float(impulse_result["threshold"])
    noise_score = float(
        np.clip(
            0.85 * robust_noise_score + 0.15 * impulse_outlier_score,
            0.0,
            100.0,
        )
    )

    uniform_typical_score = 100.0 * np.exp(-uniform_typical / 0.020)
    uniform_worst_score = 100.0 * np.exp(-uniform_worst / 0.020)
    banding_sigma = _row_column_banding_sigma(uniform_residual, flat_mask)
    banding_score = 100.0 * np.exp(-banding_sigma / 0.012)
    uniformity_score = float(
        np.clip(
            0.50 * uniform_typical_score
            + 0.30 * uniform_worst_score
            + 0.20 * banding_score,
            0.0,
            100.0,
        )
    )

    edge_strength = float(np.mean(gradient[edge_mask])) if np.any(edge_mask) else 0.0
    edge_detail = float(np.mean(detail_abs[edge_mask])) if np.any(edge_mask) else 0.0
    reference_noise = max(flat_noise_typical, 1.0 / 255.0)
    detail_snr = edge_detail / reference_noise
    sharpness_score = float(np.clip(edge_strength / 0.16 * 100.0, 0.0, 100.0))
    detail_snr_score = float(np.clip(detail_snr / 5.0 * 100.0, 0.0, 100.0))

    halo_ratio, halo_edge_sample_count = _edge_halo_overshoot_ratio(
        source_f,
        gradient_x,
        gradient_y,
        edge_mask,
        max_samples=4096,
    )
    halo_penalty = float(
        np.clip((halo_ratio - 0.10) / 0.50 * 100.0, 0.0, 100.0)
    )
    detail_clarity = float(
        np.clip(
            0.50 * sharpness_score
            + 0.50 * detail_snr_score
            - 0.08 * halo_penalty,
            0.0,
            100.0,
        )
    )

    p1, p99 = np.percentile(source_f, [1.0, 99.0])
    contrast_range = float(p99 - p1)
    entropy_bits = _entropy_bits_from_normalized(source_f)
    contrast_span_score = float(np.clip(contrast_range / 0.60 * 100.0, 0.0, 100.0))
    entropy_score = float(np.clip(entropy_bits / 7.0 * 100.0, 0.0, 100.0))
    contrast_score = float(np.clip(0.70 * contrast_span_score + 0.30 * entropy_score, 0.0, 100.0))

    clipping_percent = float(
        np.count_nonzero((source_f <= 1.0 / 255.0) | (source_f >= 254.0 / 255.0))
        * 100.0
        / source_f.size
    )
    clipping_score = float(100.0 * np.exp(-clipping_percent / 3.0))

    (
        seam_ratio,
        seam_strongest_difference_norm,
        seam_baseline_difference_norm,
        seam_peak_line_percent,
        seam_periodicity_score,
        seam_peak_group_count,
        seam_penalty,
    ) = _whole_image_seam_ratio(base)
    artifact_control = float(
        np.clip(
            0.50 * (100.0 - halo_penalty)
            + 0.50 * (100.0 - seam_penalty),
            0.0,
            100.0,
        )
    )

    overall_score = _spatial_overall_score(
        detail_clarity,
        contrast_score,
        noise_score,
        uniformity_score,
        clipping_score,
        artifact_control,
    )

    edge_region_percent = float(np.count_nonzero(edge_mask) * 100.0 / source_f.size)
    flat_region_percent = float(np.count_nonzero(flat_mask) * 100.0 / source_f.size)
    analyzed_flat_percent = float(analyzed_flat_pixels * 100.0 / source_f.size)
    flat_confidence = min(flat_region_percent / 15.0, 1.0)
    edge_confidence = min(edge_region_percent / 5.0, 1.0)
    region_confidence = min(len(flat_regions) / 3.0, 1.0)
    confidence_score = float(
        np.clip(
            (
                0.45 * np.sqrt(flat_confidence * edge_confidence)
                + 0.35 * flat_confidence
                + 0.20 * region_confidence
            )
            * 100.0,
            0.0,
            100.0,
        )
    )

    base_u8 = np.rint(np.clip(base, 0.0, 1.0) * 255.0).astype(np.uint8)
    detail_display = np.clip(detail / 0.125 * 127.0 + 128.0, 0, 255).astype(np.uint8)
    metrics = {
        "quality_profile": "RAW Full",
        "overall_score": overall_score,
        "detail_clarity_score": detail_clarity,
        "contrast_quality_score": contrast_score,
        "flat_cleanliness_score": noise_score,
        "robust_noise_score": robust_noise_score,
        "impulse_outlier_control_score": impulse_outlier_score,
        "thermal_uniformity_score": uniformity_score,
        "clipping_control_score": clipping_score,
        "artifact_control_score": artifact_control,
        "quality_confidence_score": confidence_score,
        "sharpness_score": sharpness_score,
        "detail_snr_score": detail_snr_score,
        "contrast_span_score": contrast_span_score,
        "entropy_score": entropy_score,
        "banding_control_score": float(banding_score),
        "halo_penalty_score": halo_penalty,
        "seam_artifact_penalty_score": seam_penalty,
        "detail_snr": float(detail_snr),
        "flat_noise_dn": float(flat_noise_typical * input_max),
        "flat_noise_typical_dn": float(flat_noise_typical * input_max),
        "flat_noise_worst_dn": float(flat_noise_worst * input_max),
        "flat_noise_typical_percent": float(flat_noise_typical * 100.0),
        "flat_noise_worst_percent": float(flat_noise_worst * 100.0),
        "impulse_outlier_percent": float(impulse_outlier_percent),
        "impulse_pixel_count": int(impulse_result["pixel_count"]),
        "impulse_component_count": int(impulse_result["component_count"]),
        "impulse_threshold_dn": float(impulse_threshold * input_max),
        "impulse_residual_sigma_dn": float(impulse_result["sigma"] * input_max),
        "edge_strength_dn": float(edge_strength * input_max),
        "base_residual_noise_dn": float(uniform_typical * input_max),
        "uniformity_typical_dn": float(uniform_typical * input_max),
        "uniformity_worst_dn": float(uniform_worst * input_max),
        "uniformity_typical_percent": float(uniform_typical * 100.0),
        "uniformity_worst_percent": float(uniform_worst * 100.0),
        "row_column_banding_dn": float(banding_sigma * input_max),
        "row_column_banding_percent": float(banding_sigma * 100.0),
        "halo_ratio": float(halo_ratio),
        "halo_edge_sample_count": int(halo_edge_sample_count),
        "seam_artifact_ratio": float(seam_ratio),
        "seam_periodicity_score": float(seam_periodicity_score),
        "seam_peak_group_count": int(seam_peak_group_count),
        "seam_strongest_difference_dn": float(seam_strongest_difference_norm * input_max),
        "seam_baseline_difference_dn": float(seam_baseline_difference_norm * input_max),
        "seam_peak_line_percent": float(seam_peak_line_percent),
        "p1": float(p1 * input_max),
        "p99": float(p99 * input_max),
        "p99_p1": float((p99 - p1) * input_max),
        "p99_p1_percent": float((p99 - p1) * 100.0),
        "entropy_bits": float(entropy_bits),
        "clipping_percent": clipping_percent,
        "edge_region_percent": edge_region_percent,
        "flat_region_percent": flat_region_percent,
        "flat_analyzed_region_percent": analyzed_flat_percent,
        "flat_region_group_count": int(len(flat_regions)),
        "flat_gray_band_count": int(flat_gray_band_count),
        "analysis_width": int(width),
        "analysis_height": int(height),
        "input_max": float(input_max),
    }
    return {
        "metrics": metrics,
        "base_u8": base_u8,
        "detail_u8": detail_display,
        "flat_regions": [
            {
                "band": region["band"],
                "area": region["area"],
                "mean_level_percent": round(region["mean_level"] * 100.0, 3),
                "noise_dn": round(region["noise_sigma"] * input_max, 4),
                "uniformity_dn": round(region["uniform_sigma"] * input_max, 4),
            }
            for region in flat_regions
        ],
    }
