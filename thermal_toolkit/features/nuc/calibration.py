"""features / nuc / calibration."""

import numpy as np
import os
from thermal_toolkit.core.constants import NOVATEK_NUC_GAIN_ONE, NOVATEK_NUC_GAIN_SHIFT


def calculate_nuc(low, high, min_delta):
    low_f = low.astype(np.float64)
    high_f = high.astype(np.float64)
    low_avg = low_f.mean()
    high_avg = high_f.mean()
    target_delta = high_avg - low_avg
    pixel_delta = high_f - low_f

    valid = pixel_delta > min_delta
    gain = np.ones_like(low_f, dtype=np.float64)
    offset = np.zeros_like(low_f, dtype=np.float64)
    gain[valid] = target_delta / pixel_delta[valid]
    offset[valid] = low_avg - gain[valid] * low_f[valid]
    return gain, offset, valid, pixel_delta, low_avg, high_avg


def convert_additive_b_to_gfx_b(old_b, stored_k):
    old_b_i64 = np.asarray(old_b, dtype=np.int64)
    stored_k_i64 = np.asarray(stored_k, dtype=np.int64)
    gfx_b = np.zeros(old_b_i64.shape, dtype=np.int64)
    nonzero = stored_k_i64 != 0
    numerator = -(old_b_i64[nonzero] * NOVATEK_NUC_GAIN_ONE)
    denominator = stored_k_i64[nonzero]
    rounded = (np.abs(numerator) + denominator // 2) // denominator
    gfx_b[nonzero] = np.where(numerator < 0, -rounded, rounded)
    return np.clip(gfx_b, -32768, 32767).astype(np.int16)


def build_novatek_nuc_tables(low, high, min_delta):
    low_i64 = np.asarray(low, dtype=np.int64)
    high_i64 = np.asarray(high, dtype=np.int64)
    pixel_count = low_i64.size
    low_avg = int(low_i64.sum() // pixel_count)
    high_avg = int(high_i64.sum() // pixel_count)
    target_delta = high_avg - low_avg
    if target_delta <= 0:
        raise ValueError(
            f"Novatek NUC requires HIGH average > LOW average; got LOW={low_avg}, HIGH={high_avg}."
        )

    pixel_delta = high_i64 - low_i64
    valid = pixel_delta > min_delta
    stored_k_i64 = np.full(low_i64.shape, NOVATEK_NUC_GAIN_ONE, dtype=np.int64)
    raw_k = np.zeros(low_i64.shape, dtype=np.int64)
    raw_k[valid] = (target_delta << NOVATEK_NUC_GAIN_SHIFT) // pixel_delta[valid]
    table_valid = valid & (raw_k > 0)
    stored_k_i64[table_valid] = np.clip(raw_k[table_valid], 1, 65535)

    old_b = np.zeros(low_i64.shape, dtype=np.int64)
    old_b[table_valid] = low_avg - (
        (stored_k_i64[table_valid] * low_i64[table_valid]) >> NOVATEK_NUC_GAIN_SHIFT
    )
    gfx_b = np.zeros(low_i64.shape, dtype=np.int16)
    gfx_b[table_valid] = convert_additive_b_to_gfx_b(
        old_b[table_valid], stored_k_i64[table_valid]
    )
    stored_k = stored_k_i64.astype(np.uint16)
    return stored_k, gfx_b, table_valid, low_avg, high_avg


def apply_novatek_nuc(raw, stored_k, gfx_b, raw_max, gain_shift=NOVATEK_NUC_GAIN_SHIFT):
    corrected = (
        stored_k.astype(np.int64) * (raw.astype(np.int64) - gfx_b.astype(np.int64))
    ) >> gain_shift
    return np.clip(corrected, 0, raw_max).astype(np.uint16)


def load_novatek_nuc_bin(path, width, height, dtype, table_name):
    if not path:
        raise ValueError(f"Select the Novatek {table_name} BIN file.")
    expected_bytes = width * height * np.dtype(dtype).itemsize
    actual_bytes = os.path.getsize(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{table_name} BIN size error:\n{path}\nActual: {actual_bytes:,} bytes\n"
            f"Expected: {expected_bytes:,} bytes ({width}x{height}x2)"
        )
    return np.fromfile(path, dtype=dtype).reshape((height, width))
