"""analysis / quality / uvc."""

from collections import deque
import numpy as np
import time
from thermal_toolkit.analysis.quality.metrics import (
    _entropy_bits_from_normalized,
    _isolated_impulse_analysis,
    _robust_sigma,
    _whole_image_seam_ratio,
)
from thermal_toolkit.analysis.quality.spatial import _spatial_overall_score
from thermal_toolkit.core.constants import (
    ACTIVE_UVC_TEMPORAL_WEIGHTS,
    UVC_TEMPORAL_ANALYSIS_MAX_PIXELS,
    UVC_TEMPORAL_MIN_DURATION_SECONDS,
    UVC_TEMPORAL_MIN_SAMPLES,
    UVC_TEMPORAL_SAMPLE_INTERVAL_SECONDS,
    UVC_TEMPORAL_WINDOW_SECONDS,
)
from thermal_toolkit.core.opencv import cv2


def thermal_uvc_quality_analysis(image, max_analysis_pixels=None):
    """Lightweight spatial quality analysis for live UVC streams.

    It intentionally avoids RAW Full's connected-component/gray-band segmentation and
    row/column banding pass. A downsampled single-scale Gaussian/Sobel analysis is used so
    live video output is not burdened by the quality monitor.
    """
    source_gray = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    if source_gray.ndim != 2 or source_gray.size == 0:
        raise ValueError("UVC quality analysis requires a non-empty gray image.")
    gray = source_gray
    if max_analysis_pixels and source_gray.size > int(max_analysis_pixels):
        scale = np.sqrt(float(max_analysis_pixels) / source_gray.size)
        analysis_width = max(32, int(round(source_gray.shape[1] * scale)))
        analysis_height = max(32, int(round(source_gray.shape[0] * scale)))
        gray = cv2.resize(
            source_gray,
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )

    image_f = gray.astype(np.float32) / 255.0
    base = cv2.GaussianBlur(image_f, (5, 5), 0.9)
    detail = image_f - base
    detail_abs = np.abs(detail)
    gradient_x = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y) / 4.0
    local_mean = cv2.blur(base, (7, 7))
    local_mean_square = cv2.blur(base * base, (7, 7))
    local_std = np.sqrt(np.maximum(local_mean_square - local_mean * local_mean, 0.0))

    edge_threshold = max(float(np.percentile(gradient, 75.0)), 0.015)
    edge_mask = gradient >= edge_threshold
    flat_gradient_threshold = float(
        np.clip(np.percentile(gradient, 40.0), 0.004, 0.025)
    )
    flat_std_threshold = float(
        np.clip(np.percentile(local_std, 55.0), 0.008, 0.040)
    )
    flat_mask = (gradient <= flat_gradient_threshold) & (local_std <= flat_std_threshold)
    if np.count_nonzero(flat_mask) < max(16, gray.size // 100):
        flat_mask = gradient <= flat_gradient_threshold

    edge_strength = float(np.mean(gradient[edge_mask])) if np.any(edge_mask) else 0.0
    edge_detail = float(np.mean(detail_abs[edge_mask])) if np.any(edge_mask) else 0.0
    flat_noise = _robust_sigma(detail[flat_mask]) if np.any(flat_mask) else 0.0
    detail_snr = edge_detail / max(flat_noise, 1.0 / 255.0)
    sharpness_score = float(np.clip(edge_strength / 0.16 * 100.0, 0.0, 100.0))
    detail_snr_score = float(np.clip(detail_snr / 5.0 * 100.0, 0.0, 100.0))

    # Lightweight halo proxy: high second derivative relative to the blurred edge
    # gradient rises with oversharpening/ringing, but costs only one Laplacian pass.
    laplacian = np.abs(cv2.Laplacian(image_f, cv2.CV_32F, ksize=3)) / 4.0
    halo_ratio = (
        float(np.mean(laplacian[edge_mask]))
        / max(edge_strength, 1.0 / 255.0)
        if np.any(edge_mask)
        else 0.0
    )
    halo_penalty = float(
        np.clip((halo_ratio - 0.75) / 0.80 * 100.0, 0.0, 100.0)
    )
    detail_clarity = float(
        np.clip(0.50 * sharpness_score + 0.50 * detail_snr_score - 0.08 * halo_penalty, 0.0, 100.0)
    )

    broad_base = cv2.GaussianBlur(base, (0, 0), 3.0)
    base_residual_noise = _robust_sigma((base - broad_base)[flat_mask]) if np.any(flat_mask) else 0.0
    robust_noise_score = float(100.0 * np.exp(-flat_noise / 0.025))
    impulse_result = _isolated_impulse_analysis(
        image_f,
        max_component_area=3,
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
    uniformity_score = float(100.0 * np.exp(-base_residual_noise / 0.020))

    p1, p99 = np.percentile(image_f, [1.0, 99.0])
    contrast_range = float(p99 - p1)
    entropy_bits = _entropy_bits_from_normalized(image_f)
    contrast_span_score = float(np.clip(contrast_range / 0.60 * 100.0, 0.0, 100.0))
    entropy_score = float(np.clip(entropy_bits / 7.0 * 100.0, 0.0, 100.0))
    contrast_score = float(np.clip(0.75 * contrast_span_score + 0.25 * entropy_score, 0.0, 100.0))
    clipping_percent = float(
        np.count_nonzero((gray <= 1) | (gray >= 254)) * 100.0 / gray.size
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

    base_u8 = np.rint(np.clip(base, 0.0, 1.0) * 255.0).astype(np.uint8)
    detail_display = np.clip(detail / 0.125 * 127.0 + 128.0, 0, 255).astype(np.uint8)
    output_base = base_u8
    output_detail = detail_display
    if gray.shape != source_gray.shape:
        output_base = cv2.resize(base_u8, (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        output_detail = cv2.resize(detail_display, (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    metrics = {
        "quality_profile": "UVC Lightweight",
        "overall_score": overall_score,
        "detail_clarity_score": detail_clarity,
        "contrast_quality_score": contrast_score,
        "flat_cleanliness_score": noise_score,
        "robust_noise_score": robust_noise_score,
        "impulse_outlier_control_score": impulse_outlier_score,
        "thermal_uniformity_score": uniformity_score,
        "clipping_control_score": clipping_score,
        "artifact_control_score": artifact_control,
        "sharpness_score": sharpness_score,
        "detail_snr_score": detail_snr_score,
        "contrast_span_score": contrast_span_score,
        "entropy_score": entropy_score,
        "halo_penalty_score": halo_penalty,
        "seam_artifact_penalty_score": seam_penalty,
        "detail_snr": float(detail_snr),
        "flat_noise_dn": float(flat_noise * 255.0),
        "impulse_outlier_percent": float(impulse_outlier_percent),
        "impulse_pixel_count": int(impulse_result["pixel_count"]),
        "impulse_component_count": int(impulse_result["component_count"]),
        "impulse_threshold_dn": float(impulse_threshold * 255.0),
        "impulse_residual_sigma_dn": float(impulse_result["sigma"] * 255.0),
        "edge_strength_dn": float(edge_strength * 255.0),
        "base_residual_noise_dn": float(base_residual_noise * 255.0),
        "halo_ratio": float(halo_ratio),
        "seam_artifact_ratio": float(seam_ratio),
        "seam_periodicity_score": float(seam_periodicity_score),
        "seam_peak_group_count": int(seam_peak_group_count),
        "seam_strongest_difference_dn": float(seam_strongest_difference_norm * 255.0),
        "seam_baseline_difference_dn": float(seam_baseline_difference_norm * 255.0),
        "seam_peak_line_percent": float(seam_peak_line_percent),
        "p1": float(p1 * 255.0),
        "p99": float(p99 * 255.0),
        "p99_p1": float((p99 - p1) * 255.0),
        "entropy_bits": float(entropy_bits),
        "clipping_percent": clipping_percent,
        "edge_region_percent": float(np.count_nonzero(edge_mask) * 100.0 / gray.size),
        "flat_region_percent": float(np.count_nonzero(flat_mask) * 100.0 / gray.size),
        "analysis_width": int(gray.shape[1]),
        "analysis_height": int(gray.shape[0]),
    }
    return {
        "metrics": metrics,
        "base_u8": output_base,
        "detail_u8": output_detail,
    }


class UvcTemporalQualityTracker:
    """Low-cost rolling temporal analysis for one UVC image stream."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.brightness_samples = deque()
        self.temporal_samples = deque()
        self.previous_frame = None
        self.previous_time = None
        self.frame_shape = None

    @staticmethod
    def _analysis_frame(image):
        gray = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError("Temporal analysis requires a non-empty gray image.")
        if gray.size <= UVC_TEMPORAL_ANALYSIS_MAX_PIXELS:
            return gray
        scale = np.sqrt(UVC_TEMPORAL_ANALYSIS_MAX_PIXELS / gray.size)
        width = max(32, int(round(gray.shape[1] * scale)))
        height = max(32, int(round(gray.shape[0] * scale)))
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    def add_frame(self, image, frame_time=None):
        now = time.perf_counter() if frame_time is None else float(frame_time)
        if self.previous_time is not None:
            if now <= self.previous_time:
                self.reset()
            elif now - self.previous_time < UVC_TEMPORAL_SAMPLE_INTERVAL_SECONDS:
                return False
        frame = self._analysis_frame(image)
        if self.frame_shape != frame.shape:
            self.reset()
        self.frame_shape = frame.shape

        p5, p95 = np.percentile(frame, [5.0, 95.0])
        trimmed = frame[(frame >= p5) & (frame <= p95)]
        brightness = float(np.mean(trimmed)) if trimmed.size else float(np.mean(frame))
        self.brightness_samples.append((now, brightness))

        if self.previous_frame is not None:
            current_f = frame.astype(np.float32)
            previous_f = self.previous_frame.astype(np.float32)
            difference = current_f - previous_f
            global_offset = float(np.median(difference))
            residual = difference - global_offset
            midpoint = (current_f + previous_f) * 0.5
            gradient_x = cv2.Sobel(midpoint, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(midpoint, cv2.CV_32F, 0, 1, ksize=3)
            gradient = cv2.magnitude(gradient_x, gradient_y) / 4.0
            flat_threshold = max(float(np.percentile(gradient, 60.0)), 2.0)
            flat_mask = gradient <= flat_threshold
            residual_abs = np.abs(residual)
            if np.count_nonzero(flat_mask) >= 64:
                motion_threshold = max(
                    float(np.percentile(residual_abs[flat_mask], 85.0)), 2.0
                )
                stable_mask = flat_mask & (residual_abs <= motion_threshold)
            else:
                stable_mask = residual_abs <= max(
                    float(np.percentile(residual_abs, 70.0)), 2.0
                )
            stable_count = int(np.count_nonzero(stable_mask))
            if stable_count >= 64:
                temporal_noise_dn = _robust_sigma(residual[stable_mask]) / np.sqrt(2.0)
                stable_percent = stable_count * 100.0 / frame.size
                self.temporal_samples.append(
                    (now, float(temporal_noise_dn), float(stable_percent))
                )

        self.previous_frame = frame.copy()
        self.previous_time = now
        cutoff = now - UVC_TEMPORAL_WINDOW_SECONDS
        while self.brightness_samples and self.brightness_samples[0][0] < cutoff:
            self.brightness_samples.popleft()
        while self.temporal_samples and self.temporal_samples[0][0] < cutoff:
            self.temporal_samples.popleft()
        return True

    def result(self):
        brightness = list(self.brightness_samples)
        temporal = list(self.temporal_samples)
        duration = (
            float(brightness[-1][0] - brightness[0][0])
            if len(brightness) >= 2
            else 0.0
        )
        ready = (
            duration >= UVC_TEMPORAL_MIN_DURATION_SECONDS
            and len(brightness) >= UVC_TEMPORAL_MIN_SAMPLES
            and len(temporal) >= UVC_TEMPORAL_MIN_SAMPLES - 1
        )
        result = {
            "temporal_ready": bool(ready),
            "temporal_frame_count": len(brightness),
            "temporal_window_seconds": duration,
            "flicker_dn": None,
            "temporal_noise_dn": None,
            "stable_region_percent": None,
            "flicker_control_score": None,
            "temporal_stability_score": None,
        }
        if not ready:
            return result

        times = np.asarray([item[0] for item in brightness], dtype=np.float64)
        levels = np.asarray([item[1] for item in brightness], dtype=np.float64)
        relative_times = times - times[0]
        if np.ptp(relative_times) > 0.0:
            trend = np.polyval(np.polyfit(relative_times, levels, 1), relative_times)
        else:
            trend = np.full_like(levels, np.median(levels))
        flicker_dn = float(1.4826 * np.median(np.abs(levels - trend)))
        temporal_noise_dn = float(np.median([item[1] for item in temporal]))
        stable_percent = float(np.median([item[2] for item in temporal]))
        result.update(
            {
                "flicker_dn": float(flicker_dn),
                "temporal_noise_dn": temporal_noise_dn,
                "stable_region_percent": stable_percent,
                "flicker_control_score": float(100.0 * np.exp(-flicker_dn / 2.0)),
                "temporal_stability_score": float(
                    100.0 * np.exp(-temporal_noise_dn / 3.0)
                ),
            }
        )
        return result


def add_uvc_temporal_quality(spatial_result, temporal_result):
    result = dict(spatial_result)
    result["metrics"] = dict(spatial_result["metrics"])
    metrics = result["metrics"]
    spatial_score = float(metrics["overall_score"])
    metrics["spatial_quality_score"] = spatial_score
    metrics.update(temporal_result)
    if temporal_result.get("temporal_ready"):
        metrics["overall_score"] = float(
            (
                ACTIVE_UVC_TEMPORAL_WEIGHTS["spatial_quality_score"]
                * spatial_score
                + ACTIVE_UVC_TEMPORAL_WEIGHTS["flicker_control_score"]
                * temporal_result["flicker_control_score"]
                + ACTIVE_UVC_TEMPORAL_WEIGHTS["temporal_stability_score"]
                * temporal_result["temporal_stability_score"]
            )
            / 100.0
        )
    else:
        metrics["overall_score"] = None
    return result
