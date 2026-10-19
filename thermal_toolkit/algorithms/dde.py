"""algorithms / dde."""

import numpy as np


def dde_domain_transform_u8(image, sigma_s=8, sigma_r=16,
                            gain_percent=150, num_iters=3):
    """Port of DDE_DT_Optimized (Y8, DDE_USING_MAPPING=0).

    Original Y8 differences guide every iteration. Each axis averages independent
    forward/backward recursive passes, matching the firmware's float32 order.
    """
    if image.ndim != 2 or not all(0 < size <= 4096 for size in image.shape):
        raise ValueError("DDE requires a nonempty 2D image up to 4096x4096.")
    values = (sigma_s, sigma_r, gain_percent, num_iters)
    limits = ((1, 256), (1, 255), (0, 1000), (1, 8))
    if any(not np.isfinite(v) or int(v) != v or not lo <= v <= hi
           for v, (lo, hi) in zip(values, limits)):
        raise ValueError("DDE requires integer Sigma S 1..256, Sigma R 1..255, "
                         "Gain 0..1000%, and Iterations 1..8.")
    num_iters = int(num_iters)
    source = np.ascontiguousarray(np.clip(image, 0, 255), dtype=np.uint8)
    original = source.astype(np.float32)
    base = original.copy()
    differences = [np.abs(np.diff(source.astype(np.int16), axis=axis))
                   for axis in (1, 0)]
    d = np.arange(256, dtype=np.float32)
    denom = np.sqrt(np.float32(4 ** num_iters - 1))
    for iteration in range(num_iters):
        ss = (np.float32(sigma_s) * np.sqrt(np.float32(3))
              * np.float32(2 ** (num_iters - iteration - 1)) / denom)
        ratio = ss / np.float32(sigma_r)
        factor = ratio * ratio
        coefficient = np.float32(np.sqrt(2.0) / float(ss))
        lut = np.exp(-coefficient * np.sqrt(np.float32(1) + factor * d * d))
        for axis, diff in zip((1, 0), differences):
            # Move the recursive axis first and process all independent lines together.
            src = np.moveaxis(base, axis, 0)
            weights = np.moveaxis(lut[diff], axis, 0)
            dst = np.empty_like(src)
            dst[0] = src[0]
            for i in range(1, len(src)):
                a = weights[i - 1]
                dst[i] = (np.float32(1) - a) * src[i] + a * dst[i - 1]
            backward = src[-1].copy()
            dst[-1] = (dst[-1] + backward) * np.float32(0.5)
            for i in range(len(src) - 2, -1, -1):
                a = weights[i]
                backward = (np.float32(1) - a) * src[i] + a * backward
                dst[i] = (dst[i] + backward) * np.float32(0.5)
            base = np.moveaxis(dst, 0, axis)
    gain = np.float32(gain_percent) / np.float32(100)
    enhanced = base + (original - base) * gain
    output = (np.clip(enhanced, 0, 255) + np.float32(0.5)).astype(np.uint8)
    return output, [
        "DDE Domain Transform Y8; percentile mapping disabled",
        f"Sigma S={sigma_s}, Sigma R={sigma_r}, Gain={gain_percent}%, "
        f"Iterations={num_iters}",
    ]
