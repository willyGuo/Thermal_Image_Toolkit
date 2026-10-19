"""Readable, grouped presentation of the calculated Weighted mapping values."""

from html import escape
from thermal_toolkit.algorithms.weighted_aci import selected_ratio_description, weighted_aci_statistics


def weighted_mapping_details_html(source_info, calculated, parameters):
    def heading(title):
        return f'<p style="margin-top:14px; margin-bottom:5px; color:#79c7ff;"><b>{title}</b></p>'

    def line(label, value):
        return f'<p style="margin-top:2px; margin-bottom:3px;"><b>{escape(label)}:</b> {escape(str(value))}</p>'

    parts = ['<html><body>']
    parts += [heading('1. Input / Output'),
              line('Input', f"{source_info['bit_depth']}-bit {source_info['value_name']}, 0..{source_info['value_max']}"),
              line('Actual output', f"{calculated['output_min']}..{calculated['output_max']} (8-bit)"),
              line('Grid', f"{calculated['grid']} x {calculated['grid']}")]
    sample = calculated
    if calculated['grid'] > 1:
        tiles = calculated['tiles']
        automatic = [tile for tile in tiles if 'aci_ratio' in tile]
        sample = min(automatic, key=lambda tile: tile['clip_count_per_bin']) if automatic else tiles[0]
        caps = [tile['clip_count_per_bin'] for tile in tiles if tile['clip_count_per_bin'] is not None]
        parts.append(line('Clip across tiles', f'{min(caps)}..{max(caps)} px/bin' if caps else 'Off'))
        parts.append(line('Details below', f"Tile ({sample['tile_y']}, {sample['tile_x']}); smallest automatic Clip" if automatic else 'First tile'))

    if 'aci_ratio' in sample:
        settings = sample['aci_settings']
        parts.append(heading('2. ACI Conditions'))
        if settings.get('parameter_mode', 'dual') == 'single':
            parts += [line('P99 - P1', f"{sample['aci_range']:.4f}"),
                      line('Mean', f"{sample['aci_mean']:.4f}"),
                      line('Zero mean', 'Uses 0 for (P99-P1) / mean')]
        for name, label in weighted_aci_statistics(sample):
            value, low, high = sample['aci_' + name], settings[name + '_thl'], settings[name + '_thh']
            condition = 'Low (<= THL)' if value <= low else 'High (>= THH)' if value >= high else 'Middle (THL < value < THH)'
            parts += [line(label, f'{value:.4f} — {condition}'),
                      line('Thresholds', f'THL={low:g}, THH={high:g}'),
                      line('Normalized ratio', f"clamp(({value:.4f} - {low:g}) / ({high:g} - {low:g}), 0, 1) = {sample['aci_' + name + '_ratio']:.4f}")]
        parts.append(heading('3. Automatic Clip Calculation'))
        ratio = sample['aci_ratio']
        branch = 'Low: Clip minimum' if ratio == 0 else 'High: Clip max' if ratio == 1 else 'Middle: linear Clip'
        operation, expression, driver = selected_ratio_description(sample)
        parts += [line('Ratio combination', operation),
                  line('Selected ratio', f"{expression} = {ratio:.4f}"),
                  line('Controlling condition', f'{driver}; {branch}'),
                  line('Clip max', f"2 x {sample['aci_pixel_count']:,} pixels / {sample['unique_levels']} UniqueLevels = {sample['aci_clip_max']:.3f} px/bin"),
                  line('Clip formula', f"{sample['aci_clip_min']:g} + {ratio:.4f} x ({sample['aci_clip_max']:.3f} - {sample['aci_clip_min']:g}) = {sample['aci_clip_count']:.3f}")]
    else:
        parts += [heading('2. Manual Clip'), line('Clip setting', parameters['histogram_weighted_clip'])]
    cap = sample['clip_count_per_bin']
    parts.append('<p style="margin-top:8px; margin-bottom:8px; color:#ffd479; font-size:18px;"><b>'
                 + escape(f'Actual Clip: {cap} px/bin (rounded)' if cap is not None else 'Actual Clip: Off') + '</b></p>')
    parts.append(heading('4. Mapping Settings'))
    parts += [line('Redistribution', 'On — full input DN range, including empty bins' if parameters['histogram_weighted_redistribute'] else 'Off'),
              line('Bright Weight', f"{parameters['histogram_weighted_bright_weight']:g}; threshold P{parameters['histogram_weighted_bright_percentile']:g}"),
              line('LUT Gain Limit', f"{sample['gain_limit']:g} Gray/DN" if sample['gain_limit'] else 'Off')]
    if 'aci_ratio' in sample:
        parts.append(line('Automatic mode', 'Manual Clip overridden. Selected ratio controls histogram Clip; LUT Gain Limit controls output slope separately.'))
    parts += [heading('5. Histogram Statistics'),
              line('UniqueLevels', sample['unique_levels']),
              line('Hmax / Havg', f"{sample['histogram_peak_count']:,} / {sample['histogram_average_count']:.3f}"),
              line('PeakRatio', f"{sample['peak_ratio']:.3f}"), '</body></html>']
    return ''.join(parts)
