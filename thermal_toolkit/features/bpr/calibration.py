"""DC, responsivity and NETD detection with Novatek SIE encoding."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def temporal_statistics(paths, loader):
    """Streaming population statistics: no full frame stack in memory."""
    if len(paths) < 2:
        raise ValueError("NETD requires at least 2 same-temperature RAW frames; 100 recommended.")
    mean = m2 = None
    for n, path in enumerate(paths, 1):
        frame = loader(path).astype(np.float64)
        if mean is None:
            mean, m2 = frame.copy(), np.zeros_like(frame)
        else:
            if frame.shape != mean.shape:
                raise ValueError("Temporal RAW shapes must match.")
            delta = frame - mean
            mean += delta / n
            m2 += delta * (frame - mean)
    return mean, np.sqrt(np.maximum(m2 / len(paths), 0))


def calibrate_bad_pixels(raw_a, raw_b=None, *, temperature_a=25., temperature_b=35.,
                         dc_sigma=5., response_percent=5., netd_sigma=5.,
                         window=9, temporal_std=None, dc=True, response=True, netd=False):
    """Slide criteria; DC at A, signed response, NETD in mK. Masks are unioned.

    DC uses valid neighbours only (centre excluded), population std. NETD
    reference mean/std excludes invalid responsivity; invalid slopes are flagged.
    """
    a = np.asarray(raw_a, dtype=float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("RAW A must be a non-empty 2D image.")
    if not np.isfinite(a).all():
        raise ValueError("RAW values must be finite.")
    if not any((dc, response, netd)):
        raise ValueError("Enable at least one detection method.")
    if window < 3 or window % 2 != 1 or min(dc_sigma, response_percent, netd_sigma) <= 0:
        raise ValueError("Use an odd neighbourhood >= 3 and positive thresholds.")
    reference = None
    delta = np.full(a.shape, np.nan)
    dt = temperature_b - temperature_a
    if response or netd:
        if raw_b is None:
            raise ValueError("Responsivity / NETD requires RAW B.")
        b = np.asarray(raw_b, dtype=float)
        if b.shape != a.shape or not np.isfinite(b).all():
            raise ValueError("RAW B must have the same shape as A and finite values.")
        if netd and (not np.isfinite(dt) or dt == 0):
            raise ValueError("Temperatures A and B must differ for NETD.")
        delta = b - a
        reference = float(np.median(delta))
        if reference == 0:
            raise ValueError("Median response is zero (possibly identical RAWs). Check captures.")
    radius = window // 2
    padded = np.pad(a, radius, constant_values=np.nan)
    median, std = np.empty_like(a), np.empty_like(a)
    # Bound memory while vectorizing neighbourhood statistics.
    for row in range(0, a.shape[0], 16):
        count = min(16, a.shape[0] - row)
        samples = sliding_window_view(padded[row:row + count + 2 * radius],
                                      (window, window)).reshape(count, a.shape[1], -1).copy()
        samples[:, :, window * window // 2] = np.nan
        median[row:row + count] = np.nanmedian(samples, axis=-1)
        std[row:row + count] = np.nanstd(samples, axis=-1)
    dc_difference = a - median
    response_local = np.full(a.shape, np.nan)
    deviation = np.full(a.shape, np.nan)
    response_mask = np.zeros(a.shape, dtype=bool)
    if response:
        # Fixed 9x9 response neighbourhood, independent of the DC window.
        # Exclude centre; use only valid image neighbours at borders.
        response_padded = np.pad(delta, 4, constant_values=np.nan)
        for row in range(0, a.shape[0], 16):
            count = min(16, a.shape[0] - row)
            samples = sliding_window_view(response_padded[row:row + count + 8],
                                          (9, 9)).reshape(count, a.shape[1], 81).copy()
            samples[:, :, 40] = np.nan
            response_local[row:row + count] = np.nanmedian(samples, axis=-1)
        difference = np.abs(delta - response_local)
        threshold = response_percent / 100 * np.abs(response_local)
        response_mask = difference > threshold
        np.divide(difference * 100, np.abs(response_local), out=deviation,
                  where=response_local != 0)
    masks = {'DC Level': (np.abs(dc_difference) > dc_sigma * std) & dc,
             'Responsivity': response_mask}
    values = np.full(a.shape, np.nan)
    netd_mean = netd_std = None
    masks['NETD'] = np.zeros(a.shape, dtype=bool)
    if netd:
        if temporal_std is None or temporal_std.shape != a.shape or not np.isfinite(temporal_std).all() or np.any(temporal_std < 0):
            raise ValueError("NETD requires valid same-temperature temporal statistics.")
        valid = delta * reference > 0
        values[valid] = temporal_std[valid] / np.abs(delta[valid] / dt) * 1000
        if not valid.any():
            raise ValueError("No valid pixel slopes for NETD.")
        netd_mean, netd_std = float(values[valid].mean()), float(values[valid].std())
        masks['NETD'] = ~valid | (np.abs(values - netd_mean) > netd_sigma * netd_std)
    combined = np.logical_or.reduce(list(masks.values()))
    yy, xx = np.where(combined)
    return dict(coordinates=list(zip(xx.tolist(), yy.tolist())), masks=masks,
                dc_difference=dc_difference, local_median=median, local_std=std,
                response_dn=delta, response_deviation_percent=deviation,
                response_local_reference_dn=response_local,
                response_reference_dn=reference, netd_mk=values,
                netd_mean_mk=netd_mean, netd_std_mk=netd_std)

DPC_TABLE_BYTES = 16384
DPC_MAX_PIXELS = 4095

OVERLAP_ROWS = (
    ('DC only', ('dc',)), ('Responsivity only', ('response',)), ('NETD only', ('netd',)),
    ('DC + Resp only', ('dc', 'response')), ('DC + NETD only', ('dc', 'netd')),
    ('Resp + NETD only', ('response', 'netd')), ('All three', ('dc', 'response', 'netd')),
    ('DC + Resp (incl. all 3)', ('dc', 'response')),
    ('DC + NETD (incl. all 3)', ('dc', 'netd')),
    ('Resp + NETD (incl. all 3)', ('response', 'netd')),
)


def overlap_statistics(masks):
    """Exact partitions first, followed by inclusive pair intersections."""
    named = dict(zip(('dc', 'response', 'netd'),
                     (masks['DC Level'], masks['Responsivity'], masks['NETD'])))
    counts = {}
    for i, (label, keys) in enumerate(OVERLAP_ROWS):
        selected = np.logical_and.reduce([named[k] for k in keys])
        if i < 7:
            for other in named.keys() - set(keys):
                selected &= ~named[other]
        counts[label] = int(selected.sum())
    return counts


def pixel_threshold_details(result, params, x, y):
    """CSV evidence: actual compared magnitude, threshold, and positive excess."""
    details = {}
    for key, label, unit in (('dc', 'DC Level', 'dn'),
                             ('response', 'Responsivity', 'percentage_points'),
                             ('netd', 'NETD', 'mk')):
        enabled = params[key]
        hit = bool(result['masks'][label][y, x])
        value = threshold = excess = 'N/A'
        reason = 'disabled'
        if enabled:
            if key == 'dc':
                value = abs(float(result['dc_difference'][y, x]))
                threshold = params['dc_sigma'] * float(result['local_std'][y, x])
            elif key == 'response':
                measured = float(result['response_deviation_percent'][y, x])
                value = measured if np.isfinite(measured) else 'N/A'
                threshold = params['response_percent']
            else:
                threshold = params['netd_sigma'] * result['netd_std_mk']
                if np.isfinite(result['netd_mk'][y, x]):
                    value = abs(float(result['netd_mk'][y, x]) - result['netd_mean_mk'])
            if value == 'N/A':
                reason = 'zero_local_reference' if key == 'response' else 'zero_or_opposite_response'
            else:
                excess = max(0., value - threshold)
                reason = 'above_threshold' if hit else 'within_threshold'
        details.update({f'{key}_hit': ('Yes' if hit else 'No') if enabled else 'N/A',
                        f'{key}_compared_value_{unit}': value,
                        f'{key}_threshold_{unit}': threshold,
                        f'{key}_excess_{unit}': excess,
                        f'{key}_reason': reason})
    return details


def detect_dead_pixels(raw_a, raw_b):
    """Return (x, y) candidates in row order, without scaling the input DN."""
    if raw_a.ndim != 2 or raw_a.shape != raw_b.shape or raw_a.size == 0:
        raise ValueError("The two RAW images must have the same non-empty 2D shape.")
    if np.array_equal(raw_a, raw_b):
        raise ValueError("The two RAW images are identical. Use captures at different temperatures.")
    yy, xx = np.where(raw_a == raw_b)
    return [(int(x), int(y)) for y, x in zip(yy, xx)]


def encode_dpc_table(coordinates):
    """Encode a fixed 16 KiB SIE table, rejecting overflow rather than truncating."""
    points = sorted(set(coordinates), key=lambda point: (point[1], point[0]))
    if len(points) > DPC_MAX_PIXELS:
        raise ValueError(
            f"Detected {len(points):,} candidates; Novatek supports at most "
            f"{DPC_MAX_PIXELS:,}. No table was written. Check the captures."
        )
    words = []
    current_y = None
    for x, y in points:
        if not (0 <= x < 0x8000 and 0 <= y < 0x7FFF):
            raise ValueError(f"Coordinate ({x}, {y}) cannot be encoded in SIE format.")
        if y != current_y:
            words.append(0x8000 | y)
            current_y = y
        words.append(x)
    words.extend((0xFFFF, 0x0000))
    if len(words) * 2 > DPC_TABLE_BYTES:
        raise ValueError("The SIE row markers and coordinates exceed the 16 KiB table capacity.")
    table = np.zeros(DPC_TABLE_BYTES // 2, dtype="<u2")
    table[:len(words)] = words
    return table.tobytes()
