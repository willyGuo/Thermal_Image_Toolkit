"""algorithms / enhancement."""

import numpy as np
from thermal_toolkit.algorithms.dde import dde_domain_transform_u8
from thermal_toolkit.algorithms.fieldscale import fieldscale_firmware_preview_u8
from thermal_toolkit.algorithms.mapping import percentile_preview_u8, percentile_stretch_gray, limit_percentile_gain
from thermal_toolkit.algorithms.tone_mapping import apply_tone_mapping
from thermal_toolkit.core.opencv import cv2


def apply_image_algorithm(
    gray,
    algorithm_key,
    low_percentile=1.0,
    high_percentile=99.0,
    clahe_clip=2.0,
    clahe_grid=8,
    gamma=1.5,
    median_kernel_size=3,
    gaussian_kernel_size=5,
    gaussian_sigma=1.0,
    bilateral_strength=2,
    unsharp_amount=1.0,
    clahe=None,
    input_max=255,
    fieldscale_state=None,
    fieldscale_temporal=False,
    grid_count=8,
    les_distance=2,
    les_threshold_raw14=400,
    message_passing_distance=1,
    message_passing_iterations=7,
    temporal_smoothing_percent=90,
    max_local_gain=0.0,
    dde_sigma_s=21,
    dde_sigma_r=51,
    dde_gain_percent=200,
    dde_num_iters=3,
    histogram_weighted_grid=1,
    histogram_weighted_clip=2.0,
    histogram_weighted_redistribute=True,
    histogram_weighted_bright_weight=1.0,
    histogram_weighted_bright_percentile=80.0,
    histogram_weighted_gain_limit=0,
    percentile_gain_limit=0.0,
):
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. pip install opencv-python")
    if gray.ndim != 2:
        raise ValueError("Image algorithm input must be a 2D grayscale image.")

    details = []

    if algorithm_key == "raw_percentile_preview":
        processed, vmin, vmax = percentile_preview_u8(gray)
        processed, _ = limit_percentile_gain(gray, processed, vmin, vmax, percentile_gain_limit)
        details.append(f"Original-value P1.0={vmin:.1f}, P99.0={vmax:.1f}")
        return processed, details

    if algorithm_key == "fieldscale":
        return fieldscale_firmware_preview_u8(
            gray,
            input_max,
            state=fieldscale_state,
            temporal=fieldscale_temporal,
            grid_count=grid_count,
            les_distance=les_distance,
            les_threshold_raw14=les_threshold_raw14,
            message_passing_distance=message_passing_distance,
            message_passing_iterations=message_passing_iterations,
            temporal_smoothing_percent=temporal_smoothing_percent,
            max_local_gain=max_local_gain,
        )

    if algorithm_key == "dde":
        return dde_domain_transform_u8(
            gray,
            sigma_s=dde_sigma_s,
            sigma_r=dde_sigma_r,
            gain_percent=dde_gain_percent,
            num_iters=dde_num_iters,
        )

    processed = np.ascontiguousarray(np.clip(gray, 0, 255).astype(np.uint8))

    if algorithm_key == "histogram_weighted":
        processed, mapping_details, _, _ = apply_tone_mapping(
            processed, "histogram_weighted", 255, {
                "histogram_weighted_grid": histogram_weighted_grid,
                "histogram_weighted_clip": histogram_weighted_clip,
                "histogram_weighted_redistribute": histogram_weighted_redistribute,
                "histogram_weighted_bright_weight": histogram_weighted_bright_weight,
                "histogram_weighted_bright_percentile": histogram_weighted_bright_percentile,
                "histogram_weighted_gain_limit": histogram_weighted_gain_limit,
            },
        )
        return processed, ["8-bit enhancement input (0..255)", mapping_details]

    if algorithm_key in ("auto_contrast", "auto_contrast_clahe"):
        percentile_input = gray if input_max > 255 else processed
        if input_max > 255:
            processed, vmin, vmax = percentile_preview_u8(
                gray, low_percentile, high_percentile
            )
        else:
            processed, vmin, vmax = percentile_stretch_gray(
                processed, low_percentile, high_percentile
            )
        details.append(
            f"P{low_percentile:.1f}={vmin:.1f}, P{high_percentile:.1f}={vmax:.1f}"
        )
        processed, gain_info = limit_percentile_gain(
            percentile_input, processed, vmin, vmax, percentile_gain_limit
        )
        details.append(
            f"Gain Limit={gain_info['gain_limit']:g} Gray/DN; "
            f"applied={'yes' if gain_info['gain_limit_applied'] else 'no'}; "
            "preserve lower-median brightness; no rescaling; round to 8-bit"
            if gain_info["gain_limit"] else "Gain Limit=Off"
        )

    if algorithm_key in ("clahe", "auto_contrast_clahe"):
        if clahe is None:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip,
                tileGridSize=(clahe_grid, clahe_grid),
            )
        processed = clahe.apply(processed)
        details.append(f"CLAHE clip={clahe_clip:.1f}, grid={clahe_grid}x{clahe_grid}")

    if algorithm_key == "hist_equalization":
        processed = cv2.equalizeHist(processed)
        details.append("Global histogram equalization")
    elif algorithm_key == "gamma":
        lut = np.array(
            [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
            dtype=np.uint8,
        )
        processed = cv2.LUT(processed, lut)
        details.append(f"Gamma={gamma:.1f}")
    elif algorithm_key == "median":
        median_kernel_size = int(median_kernel_size)
        if median_kernel_size < 3 or median_kernel_size % 2 == 0:
            raise ValueError("Median Kernel Size must be an odd integer of at least 3.")
        processed = cv2.medianBlur(processed, median_kernel_size)
        details.append(f"Median kernel={median_kernel_size}x{median_kernel_size}")
    elif algorithm_key == "gaussian":
        gaussian_kernel_size = int(gaussian_kernel_size)
        gaussian_sigma = float(gaussian_sigma)
        if gaussian_kernel_size < 3 or gaussian_kernel_size % 2 == 0:
            raise ValueError("Gaussian Kernel Size must be an odd integer of at least 3.")
        if gaussian_sigma <= 0.0:
            raise ValueError("Gaussian Sigma must be greater than 0.")
        processed = cv2.GaussianBlur(
            processed,
            (gaussian_kernel_size, gaussian_kernel_size),
            gaussian_sigma,
        )
        details.append(
            f"Gaussian kernel={gaussian_kernel_size}x{gaussian_kernel_size}, "
            f"sigma={gaussian_sigma:.1f}"
        )
    elif algorithm_key == "bilateral":
        diameter = bilateral_strength * 2 + 1
        sigma = bilateral_strength * 20
        processed = cv2.bilateralFilter(processed, diameter, sigma, sigma)
        details.append(f"Bilateral strength={bilateral_strength}")
    elif algorithm_key == "unsharp":
        blurred = cv2.GaussianBlur(processed, (0, 0), 1.0)
        processed = cv2.addWeighted(
            processed, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0
        )
        details.append(f"Unsharp amount={unsharp_amount:.1f}")

    if not details:
        raise ValueError(f"Unknown image algorithm: {algorithm_key}")
    return processed, details
