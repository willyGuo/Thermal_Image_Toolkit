"""analysis / statistics."""

import numpy as np


def image_stats(image):
    arr = np.asarray(image, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p1": float(np.percentile(arr, 1.0)),
        "p99": float(np.percentile(arr, 99.0)),
    }


def gray_image_analysis(image):
    gray = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    flat = gray.reshape(-1)
    histogram = np.bincount(flat, minlength=256).astype(np.int64)
    percentiles = np.percentile(flat, [1.0, 5.0, 50.0, 95.0, 99.0])
    probabilities = histogram[histogram > 0].astype(np.float64) / flat.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    zero_count = int(histogram[0])
    full_count = int(histogram[255])
    limited_low_count = int(histogram[:17].sum())
    limited_high_count = int(histogram[235:].sum())
    stats = {
        "count": int(flat.size),
        "min": int(flat.min()),
        "max": int(flat.max()),
        "mean": float(flat.mean()),
        "median": float(percentiles[2]),
        "std": float(flat.std()),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "p99_p1": float(percentiles[4] - percentiles[0]),
        "unique_levels": int(np.count_nonzero(histogram)),
        "entropy_bits": entropy,
        "zero_count": zero_count,
        "zero_percent": float(zero_count * 100.0 / flat.size),
        "full_count": full_count,
        "full_percent": float(full_count * 100.0 / flat.size),
        "limited_low_count": limited_low_count,
        "limited_low_percent": float(limited_low_count * 100.0 / flat.size),
        "limited_high_count": limited_high_count,
        "limited_high_percent": float(limited_high_count * 100.0 / flat.size),
    }
    return stats, histogram


def raw_dn_image_analysis(image, raw_max):
    raw = np.asarray(image).astype(np.uint16, copy=False)
    flat = raw.reshape(-1)
    percentiles = np.percentile(flat, [1.0, 5.0, 50.0, 95.0, 99.0])
    _levels, level_counts = np.unique(flat, return_counts=True)
    probabilities = level_counts.astype(np.float64) / flat.size
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    histogram, _edges = np.histogram(
        np.clip(flat, 0, raw_max),
        bins=256,
        range=(0, int(raw_max) + 1),
    )
    zero_count = int(np.count_nonzero(flat == 0))
    max_count = int(np.count_nonzero(flat == raw_max))
    over_max_count = int(np.count_nonzero(flat > raw_max))
    p99_p1 = float(percentiles[4] - percentiles[0])
    stats = {
        "count": int(flat.size),
        "min": int(flat.min()),
        "max": int(flat.max()),
        "mean": float(flat.mean()),
        "median": float(percentiles[2]),
        "std": float(flat.std()),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "p99_p1": p99_p1,
        "p99_p1_percent": float(p99_p1 * 100.0 / raw_max),
        "unique_levels": int(level_counts.size),
        "entropy_bits": entropy,
        "zero_count": zero_count,
        "zero_percent": float(zero_count * 100.0 / flat.size),
        "max_count": max_count,
        "max_percent": float(max_count * 100.0 / flat.size),
        "over_max_count": over_max_count,
        "over_max_percent": float(over_max_count * 100.0 / flat.size),
        "raw_max": int(raw_max),
    }
    return stats, histogram.astype(np.int64)
