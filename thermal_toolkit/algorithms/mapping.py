"""algorithms / mapping."""

import numpy as np
from thermal_toolkit.core.opencv import cv2


def raw_to_display_u8(image, raw_max):
    scaled = np.clip(image, 0, raw_max).astype(np.float64) * 255.0 / raw_max
    return np.rint(scaled).astype(np.uint8)


def signed_delta_to_display_u8(image, raw_max):
    scaled = (
        np.clip(image, -raw_max, raw_max).astype(np.float64) + raw_max
    ) * 255.0 / (2.0 * raw_max)
    return np.rint(scaled).astype(np.uint8)


def percentile_stretch_gray(gray, low_p=1.0, high_p=99.0):
    gray_f = gray.astype(np.float32)
    vmin = float(np.percentile(gray_f, low_p))
    vmax = float(np.percentile(gray_f, high_p))
    if vmax <= vmin:
        return gray.copy(), vmin, vmax
    stretched = (gray_f - vmin) * 255.0 / (vmax - vmin)
    return np.clip(stretched, 0, 255).astype(np.uint8), vmin, vmax


def percentile_preview_u8(image, p_low=1.0, p_high=99.0):
    arr = image.astype(np.float64)
    vmin = float(np.percentile(arr, p_low))
    vmax = float(np.percentile(arr, p_high))
    if vmax <= vmin:
        vmin = float(arr.min())
        vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8), vmin, vmax
    stretched = (arr - vmin) * 255.0 / (vmax - vmin)
    return np.clip(stretched, 0, 255).astype(np.uint8), vmin, vmax


def limit_percentile_gain(image, mapped, low_dn, high_dn, gain_limit=0.0):
    """Cap percentile slope before uint8 rounding, anchored at lower-median brightness.

    Keep percentile clipping and the existing output when disabled or below the cap.
    Constant-input fallback behavior belongs to the original mapping function.
    """
    gain_limit = float(gain_limit)
    if (not np.isfinite(gain_limit) or not 0.0 <= gain_limit <= 4.0
            or not np.isclose(gain_limit * 10, round(gain_limit * 10), rtol=0, atol=1e-9)):
        raise ValueError("Percentile Gain Limit must be 0.0 (Off) to 4.0 Gray/DN in steps of 0.1.")
    gain_limit = round(gain_limit, 1)
    original_gain = 255.0 / (high_dn - low_dn) if high_dn > low_dn else None
    applied = bool(gain_limit and original_gain is not None and original_gain > gain_limit)
    info = {
        "gain_limit": gain_limit, "gain_limit_applied": applied,
        "original_gain": original_gain,
        "effective_gain": gain_limit if applied else original_gain,
        "gain_limit_domain": "before_output_rounding",
        "gain_anchor_dn": None, "gain_anchor_gray": None,
    }
    if not applied:
        return mapped, info
    source = np.asarray(image, dtype=np.float64)
    flat = source.ravel()
    rank = (flat.size - 1) // 2
    anchor_index = int(np.argpartition(flat, rank)[rank])
    anchor_dn = float(flat[anchor_index])
    anchor_gray = int(mapped.flat[anchor_index])
    clipped = np.clip(source, low_dn, high_dn)
    limited = anchor_gray + (clipped - np.clip(anchor_dn, low_dn, high_dn)) * gain_limit
    output = np.floor(np.clip(limited, 0, 255) + 0.5).astype(np.uint8)
    info.update(gain_anchor_dn=anchor_dn, gain_anchor_gray=anchor_gray)
    return output, info


def uvc_to_gray(frame):
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
