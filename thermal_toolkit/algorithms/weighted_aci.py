"""Shared runtime configuration for histogram-weighted automatic clipping."""

import numpy as np


def default_weighted_aci_settings():
    return dict(enabled=True, range_thl=75.0, range_thh=350.0,
                sigma_thl=18.0, sigma_thh=121.0, ratio_mode='min',
                parameter_mode='dual', relative_range_thl=0.0, relative_range_thh=1.0)


def validate_weighted_aci_settings(candidate):
    result = default_weighted_aci_settings()
    if not isinstance(candidate, dict):
        return result
    result['enabled'] = bool(candidate.get('enabled', True))
    mode = candidate.get('ratio_mode', 'min')
    result['ratio_mode'] = mode if mode in ('min', 'mean', 'max') else 'min'
    parameter_mode = candidate.get('parameter_mode', 'dual')
    result['parameter_mode'] = parameter_mode if parameter_mode in ('single', 'dual') else 'dual'
    for name in ('range', 'sigma', 'relative_range'):
        try:
            low = float(candidate.get(name + '_thl', result[name + '_thl']))
            high = float(candidate.get(name + '_thh', result[name + '_thh']))
            if not (np.isfinite(low) and np.isfinite(high) and 0 <= low < high <= 65535):
                raise ValueError('THL must be smaller than THH.')
        except (TypeError, ValueError, OverflowError):
            continue
        result[name + '_thl'], result[name + '_thh'] = low, high
    return result


# Library callers retain manual clipping unless the application applies Settings.
ACTIVE_WEIGHTED_ACI = dict(default_weighted_aci_settings(), enabled=False)


def configure_weighted_aci(settings):
    global ACTIVE_WEIGHTED_ACI
    ACTIVE_WEIGHTED_ACI = validate_weighted_aci_settings(settings)


def weighted_aci_clip(source, unique_levels):
    settings = dict(ACTIVE_WEIGHTED_ACI)
    if not settings['enabled'] or not source.size or unique_levels < 1:
        return None
    p1, p99 = np.percentile(source, [1, 99])
    spread, sigma = float(p99 - p1), float(np.std(source, dtype=np.float64))
    ratios = [float(np.clip((value - settings[name + '_thl']) /
                           (settings[name + '_thh'] - settings[name + '_thl']), 0, 1))
              for name, value in (('range', spread), ('sigma', sigma))]
    mean = float(np.mean(source, dtype=np.float64))
    # Thermal inputs are nonnegative; a zero-mean tile is all black.
    relative_range = spread / mean if mean > 0 else 0.0
    relative_ratio = float(np.clip(
        (relative_range - settings['relative_range_thl']) /
        (settings['relative_range_thh'] - settings['relative_range_thl']), 0, 1))
    maximum = 2.0 * source.size / unique_levels
    # Small images/tiles can have Clip max below 20; keep the cap bounded.
    minimum = min(20.0, maximum)
    ratio = (sum(ratios) / 2 if settings['ratio_mode'] == 'mean' else
             max(ratios) if settings['ratio_mode'] == 'max' else min(ratios))
    if settings['parameter_mode'] == 'single':
        ratio = relative_ratio
    return dict(aci_settings=settings, aci_range=spread, aci_sigma=sigma,
                aci_mean=mean, aci_relative_range=relative_range,
                aci_relative_range_ratio=relative_ratio,
                aci_range_ratio=ratios[0], aci_sigma_ratio=ratios[1],
                aci_ratio=ratio, aci_clip_max=maximum, aci_clip_min=minimum,
                aci_pixel_count=int(source.size),
                aci_clip_count=minimum + ratio * (maximum - minimum))


def selected_ratio_description(info):
    """Return the selected operation, expression and controlling statistic."""
    if info['aci_settings'].get('parameter_mode', 'dual') == 'single':
        return 'Single parameter', f"{info['aci_relative_range_ratio']:.4f}", '(P99-P1) / mean'
    mode = info['aci_settings'].get('ratio_mode', 'min')
    a, b = info['aci_range_ratio'], info['aci_sigma_ratio']
    if mode == 'mean':
        return 'Average', f'({a:.4f} + {b:.4f}) / 2', 'average of range and standard deviation'
    driver = ('both' if a == b else 'range' if
              (a < b if mode == 'min' else a > b) else 'standard deviation')
    return mode.upper(), f'{mode}({a:.4f}, {b:.4f})', driver


def weighted_aci_statistics(info):
    """Statistics used by the selected automatic Clip mode."""
    if info['aci_settings'].get('parameter_mode', 'dual') == 'single':
        return (('relative_range', '(P99-P1) / mean'),)
    return (('range', 'Range (P99-P1)'), ('sigma', 'Standard deviation'))


def describe_weighted_aci(info):
    """Explain the measured conditions and actual cap from this LUT's snapshot."""
    settings = info['aci_settings']
    lines = ["Automatic Clip conditions:"]
    if settings.get('parameter_mode', 'dual') == 'single':
        lines.append(f"(P99-P1)={info['aci_range']:.4f}; mean={info['aci_mean']:.4f}; "
                     f"(P99-P1) / mean={info['aci_relative_range']:.4f} (zero mean uses 0).")
    for name, label in weighted_aci_statistics(info):
        value = info['aci_' + name]
        low, high = settings[name + '_thl'], settings[name + '_thh']
        condition = (f"<= THL {low:g} (low)" if value <= low else
                     f">= THH {high:g} (high)" if value >= high else
                     f"between THL {low:g} and THH {high:g} (middle)")
        lines.append(f"{label}={value:.4f}: {condition}; "
                     f"ratio=clamp(({value:.4f}-{low:g})/({high:g}-{low:g}), 0, 1)"
                     f"={info['aci_' + name + '_ratio']:.4f}.")
    ratio = info['aci_ratio']
    operation, expression, driver = selected_ratio_description(info)
    branch = 'low: use Clip minimum' if ratio == 0 else 'high: use Clip max' if ratio == 1 else 'middle: linear Clip'
    lines.append(f"Selected ratio ({operation})={expression}={ratio:.4f}, controlled by {driver}; {branch}.")
    maximum, minimum = info['aci_clip_max'], info['aci_clip_min']
    lines.append(f"Clip max=2 * {info['aci_pixel_count']:,} pixels / "
                 f"{info['unique_levels']} UniqueLevels={maximum:.3f} px/bin.")
    lines.append(f"Clip={minimum:g} + {ratio:.4f} * ({maximum:.3f}-{minimum:g})"
                 f"={info['aci_clip_count']:.3f} -> {info['clip_count_per_bin']} px/bin (rounded); manual Clip overridden.")
    lines.append("The selected ratio controls histogram Clip; LUT Gain Limit below is a separate output-slope limit.")
    return '\n'.join(lines)
