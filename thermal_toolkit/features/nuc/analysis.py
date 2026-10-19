"""features / nuc / analysis."""

import numpy as np
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.features.nuc.definitions import NUC_COMPARISON_ROIS, NUC_ROI_METRICS


def nuc_analysis_statistics(image, raw_max):
    """Spatial metrics matching the 20260921 analyze.py reference."""
    a = np.asarray(image, dtype=np.float64)
    height, width = a.shape
    ch, cw = min(256, height), min(256, width)
    y, x = (height - ch) // 2, (width - cw) // 2
    center = a[y:y + ch, x:x + cw]
    rim = np.ones(a.shape, dtype=bool)
    rim[32:-32, 32:-32] = False
    bh, bw = height // 64, width // 64
    blocks = a[:bh * 64, :bw * 64].reshape(bh, 64, bw, 64).mean(axis=(1, 3)) if bh and bw else None
    # The reference removes a two-pixel border, so median border modes do not affect results.
    residual = a - cv2.medianBlur(np.ascontiguousarray(image, dtype=np.uint16), 3).astype(np.float64)
    residual = residual[2:-2, 2:-2]
    sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual)))) if residual.size else None
    threshold = max(8 * sigma, 10.0) if sigma is not None else None
    stats = {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
             "max": float(a.max()), "peak_to_peak": float(np.ptp(a)), "median": float(np.median(a)),
             "p1_DN": float(np.percentile(a, 1)), "p99_DN": float(np.percentile(a, 99)),
             "unique_levels": int(np.unique(a).size), "zero_pixels": int(np.count_nonzero(a == 0)),
             "fullscale_pixels": int(np.count_nonzero(a == raw_max)),
             "center_mean_DN": float(center.mean()), "center_std_DN": float(center.std()),
             "rim_mean_DN": float(a[rim].mean()),
             "rim_center_difference_DN": float(a[rim].mean() - center.mean()),
             "row_mean_std_DN": float(a.mean(axis=1).std()),
             "column_mean_std_DN": float(a.mean(axis=0).std()),
             "block64_mean_std_DN": float(blocks.std()) if blocks is not None else None,
             "block64_mean_range_DN": float(np.ptp(blocks)) if blocks is not None else None,
             "local_residual_sigma_DN": sigma, "local_candidate_threshold_DN": threshold,
             "local_candidate_pixels": int(np.count_nonzero(np.abs(residual - np.median(residual)) > threshold)) if residual.size else None}
    return stats


def nuc_roi_analysis(raw, corrected, stored_k, gfx_b, raw_max, n):
    regions = {}
    for name, (x0, y0, x1, y1) in NUC_COMPARISON_ROIS.items():
        valid = x1 <= raw.shape[1] and y1 <= raw.shape[0]
        result = {"xyxy": [x0, y0, x1, y1], "valid": valid,
                  "pixel_count": (x1 - x0) * (y1 - y0) if valid else 0}
        if valid:
            roi = np.s_[y0:y1, x0:x1]
            result["original"] = {k: v for k, v in nuc_analysis_statistics(raw[roi], raw_max).items() if k in NUC_ROI_METRICS}
            result["corrected"] = {k: v for k, v in nuc_analysis_statistics(corrected[roi], raw_max).items() if k in NUC_ROI_METRICS}
            unbounded = (stored_k[roi].astype(np.int64) * (raw[roi].astype(np.int64) - gfx_b[roi].astype(np.int64))) >> n
            result["clipped_low_count"] = int(np.count_nonzero(unbounded < 0))
            result["clipped_high_count"] = int(np.count_nonzero(unbounded > raw_max))
        regions[name] = result
    comparison = {}
    if all(region["valid"] for region in regions.values()):
        for stage in ("original", "corrected"):
            left = regions["左側參考"][stage]["mean"]
            right = regions["右側參考"][stage]["mean"]
            reference = (left + right) / 2
            comparison[stage] = {"reference_mean_DN": reference,
                                 "block_minus_reference_DN": regions["方塊"][stage]["mean"] - reference}
    reference_names = ("左側參考", "右側參考")
    combined = {"xyxy": None, "source_rois": [regions[name]["xyxy"] for name in reference_names],
                "valid": all(regions[name]["valid"] for name in reference_names), "pixel_count": 0}
    if combined["valid"]:
        combined["pixel_count"] = sum(regions[name]["pixel_count"] for name in reference_names)
        for stage, image in (("original", raw), ("corrected", corrected)):
            parts = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in combined["source_rois"]]
            stats = nuc_analysis_statistics(np.concatenate(parts, axis=1), raw_max)
            combined[stage] = {key: stats[key] for key in NUC_ROI_METRICS}
        for key in ("clipped_low_count", "clipped_high_count"):
            combined[key] = sum(regions[name][key] for name in reference_names)
    regions["左右合併參考"] = combined
    for region in regions.values():
        region["reduction"] = {}
        for key in NUC_ROI_METRICS:
            before = region.get("original", {}).get(key)
            after = region.get("corrected", {}).get(key)
            decrease = before - after if before is not None else None
            region["reduction"][key] = {"amount": decrease, "percent": decrease / before * 100 if before not in (None, 0) else None}
    if comparison:
        for stage in ("original", "corrected"):
            comparison[stage]["absolute_block_difference_DN"] = abs(comparison[stage]["block_minus_reference_DN"])
        comparison["reduction"] = {}
        for key, before in comparison["original"].items():
            decrease = before - comparison["corrected"][key]
            comparison["reduction"][key] = {"amount": decrease, "percent": decrease / before * 100 if before != 0 else None}
    return {"regions": regions, "comparison": comparison,
            "coordinate_system": "RAW top-left zero-based; end exclusive",
            "comparison_definition": "左右平均 = (左側平均 + 右側平均) / 2；合併統計使用左右兩區全部像素，不含中間方塊，標準差不是兩區標準差的平均。方塊減左右平均，正值較亮、負值較暗；亮度差改善以差值絕對值的下降量及下降率判讀。"}
