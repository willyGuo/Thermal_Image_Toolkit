"""algorithms / tone_mapping."""

import numpy as np
from thermal_toolkit.algorithms.fieldscale import FIELDSCALE_RAW14_MAX, fieldscale_firmware_preview_u8
from thermal_toolkit.algorithms.histogram import (
    clahe_unique_levels_to_8bit,
)
from thermal_toolkit.algorithms.mapping import percentile_preview_u8, limit_percentile_gain
from thermal_toolkit.algorithms.parameters import DEFAULT_TONE_MAPPING_PARAMETERS
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.algorithms.weighted_grid import histogram_weighted_grid_u8
from thermal_toolkit.algorithms import weighted_aci
from thermal_toolkit.algorithms.weighted_plateau import weighted_plateau_u8


def tone_mapping_parameters(method, parameters=None, input_max=255):
    if method not in DEFAULT_TONE_MAPPING_PARAMETERS:
        raise ValueError(f"Unsupported tone mapping method: {method}")
    result = dict(DEFAULT_TONE_MAPPING_PARAMETERS[method])
    if method == "linear":
        result["white_dn"] = int(input_max)
    if parameters:
        result.update(parameters)
    if method == "weighted_plateau":
        # Ignore removed mode/range settings from older saved parameter snapshots.
        result.pop("plateau_mode", None)
        result.pop("plateau_low_percentile", None)
        result.pop("plateau_high_percentile", None)
    return result


def apply_tone_mapping(
    image,
    method,
    input_max,
    parameters=None,
    pad_fieldscale_input=False,
):
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. pip install opencv-python")
    if image.ndim != 2:
        raise ValueError("Tone mapping input must be a 2D grayscale image.")
    input_max = max(1, int(input_max))
    params = tone_mapping_parameters(method, parameters, input_max)
    calculated = {"input_min": int(np.min(image)), "input_max": int(np.max(image))}

    if method == "linear":
        black_dn = int(params["black_dn"])
        white_dn = int(params["white_dn"])
        if not 0 <= black_dn < white_dn <= input_max:
            raise ValueError(
                f"Linear mapping requires 0 <= Black DN < White DN <= {input_max}."
            )
        mapped = np.clip(
            (image.astype(np.float64) - black_dn) * 255.0 / (white_dn - black_dn),
            0,
            255,
        ).astype(np.uint8)
        calculated.update({"black_dn": black_dn, "white_dn": white_dn})
        details = f"Linear {black_dn}..{white_dn} DN -> 8-bit"
    elif method == "percentile":
        low = float(params["low_percentile"])
        high = float(params["high_percentile"])
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("Percentile mapping requires Low Percentile < High Percentile.")
        mapped, low_dn, high_dn = percentile_preview_u8(image, low, high)
        mapped, gain_info = limit_percentile_gain(
            image, mapped, low_dn, high_dn, params["percentile_gain_limit"]
        )
        calculated.update({"low_dn": low_dn, "high_dn": high_dn})
        calculated.update(gain_info)
        calculated.update(output_min=int(mapped.min()), output_max=int(mapped.max()))
        details = f"P{low:.1f}={low_dn:.1f}, P{high:.1f}={high_dn:.1f} DN -> 8-bit"
        details += (
            f"; Gain Limit={gain_info['gain_limit']:g} Gray/DN; "
            f"applied={'yes' if gain_info['gain_limit_applied'] else 'no'}; "
            "preserve lower-median brightness; no rescaling; round to 8-bit"
            if gain_info["gain_limit"] else "; Gain Limit=Off"
        )
    elif method == "weighted_plateau":
        mapped, plateau_info = weighted_plateau_u8(image, input_max, **params)
        calculated.update(plateau_info)
        details = (
            "Global Double Plateau: "
            f"DN {plateau_info['mapping_first_dn']}..{plateau_info['mapping_last_dn']}; "
            f"Unique Levels={plateau_info['unique_levels']}; "
            f"average={plateau_info['histogram_average_count']:.3f} px/populated bin\n"
            f"Upper={params['plateau_upper']:g} x average -> {plateau_info['upper_plateau_count']} px/bin; "
            f"Lower={params['plateau_lower']:g} x average"
            f" -> {plateau_info['lower_plateau_count']} px/bin\n"
            f"Redistribution={'populated bins only' if params['plateau_redistribute'] else 'Off'}; "
            f"clipped={plateau_info['clipped_histogram_count']:,}; "
            f"redistributed={plateau_info['redistributed_histogram_count']:,}; "
            f"discarded={plateau_info['discarded_histogram_count']:,}; "
            f"floor added={plateau_info['floor_added_histogram_count']:,}\n"
            f"W(i)={plateau_info['weight_formula']}; cumulative interval mapping; "
            "normalize over observed input min/max."
        )
        if plateau_info['constant_input_range']:
            details += " Constant input: output 128."
        details += (
            f"\nBright threshold P{plateau_info['bright_percentile']:g}="
            f"{plateau_info['bright_threshold_dn']:.2f} DN (original input); "
            f"Bright Weight={plateau_info['bright_weight']:g}; "
            "multiply intervals at/above threshold before normalization and Gain Limit."
        )
        if plateau_info['gain_limit']:
            details += (
                f"\nLUT Gain Limit={plateau_info['gain_limit']:g} Gray/DN; "
                f"anchor DN {plateau_info['gain_anchor_dn']} -> {plateau_info['gain_anchor_gray']}; "
                "preserve original lower-median brightness; no rescaling; "
                "fractional limits apply before 8-bit rounding."
            )
        else:
            details += "\nLUT Gain Limit=Off"
    elif method == "histogram_weighted":
        clip_limit = float(params["histogram_weighted_clip"])
        redistribute = bool(params["histogram_weighted_redistribute"])
        mapped, weighted_info = histogram_weighted_grid_u8(
            image, input_max, clip_limit, redistribute,
            params["histogram_weighted_bright_weight"],
            params["histogram_weighted_bright_percentile"],
            grid=params["histogram_weighted_grid"],
            gain_limit=params["histogram_weighted_gain_limit"],
        )
        calculated.update(weighted_info)
        calculated["redistribute_clipped_pixels"] = redistribute
        details = (
            f"Histogram-weighted grid={weighted_info['grid']}x{weighted_info['grid']}; "
            f"bilinear tile-LUT interpolation; clip={clip_limit:g} x tile pixels / "
            f"tile Unique Levels; redistribute={'on' if redistribute else 'off'}; "
            f"per-tile P{params['histogram_weighted_bright_percentile']:g}, "
            f"bright weight={params['histogram_weighted_bright_weight']:g}"
        ) if weighted_info["grid"] > 1 else (
            "Histogram-weighted mapping: W(i)=H(i)+H(i+1), normalize all "
            f"interval weights to 0..255 before gain limiting; "
            f"DN {weighted_info['first_populated_dn']} -> {weighted_info['first_output_gray']}, "
            f"DN {weighted_info['last_populated_dn']} -> {weighted_info['last_output_gray']}; "
            + (
                ("automatic histogram Clip, " if 'aci_ratio' in weighted_info else
                 f"clip={clip_limit:.1f} (pixel count / Unique Levels), ") +
                f"internal clip={weighted_info['clip_count_per_bin']} px/bin; "
                f"redistribute={'on' if redistribute else 'off'}"
                if clip_limit > 0.0 or 'aci_ratio' in weighted_info
                else "clip=off"
            )
            + (
                f"; P{weighted_info['bright_percentile']:g}="
                f"{weighted_info['bright_threshold_dn']:.2f} DN, "
                f"bright weight={weighted_info['bright_weight']:g}; "
                f"Hmax={weighted_info['histogram_peak_count']:,}, "
                f"Havg={weighted_info['histogram_average_count']:.3f}, "
                f"PeakRatio={weighted_info['peak_ratio']:.3f}"
            )
        )
        if 'aci_ratio' in weighted_info:
            details += '\n' + weighted_aci.describe_weighted_aci(weighted_info)
        elif weighted_info['grid'] > 1:
            tiles = [tile for tile in weighted_info['tiles'] if 'aci_ratio' in tile]
            if tiles:
                caps = [tile['clip_count_per_bin'] for tile in tiles]
                counts = [sum(tile['aci_ratio'] == 0 for tile in tiles),
                          sum(0 < tile['aci_ratio'] < 1 for tile in tiles),
                          sum(tile['aci_ratio'] == 1 for tile in tiles)]
                details += (f"\nAutomatic Clip per tile: low/middle/high={counts[0]}/{counts[1]}/{counts[2]}; "
                            f"actual Clip={min(caps)}..{max(caps)} px/bin; manual Clip overridden.")
                sample = min(tiles, key=lambda tile: tile['clip_count_per_bin'])
                details += (f"\nTile ({sample['tile_y']}, {sample['tile_x']}) with smallest Clip:\n"
                            + weighted_aci.describe_weighted_aci(sample))
        if 'aci_ratio' in weighted_info or (weighted_info['grid'] > 1 and any('aci_ratio' in tile for tile in weighted_info['tiles'])):
            details += ('\nRedistribution: full input DN range, including empty bins.' if redistribute else
                        '\nRedistribution: Off; clipped excess discarded, populated-range mapping.')
        gain_limit = weighted_info["gain_limit"]
        details += (
            f"\nLUT Gain Limit={gain_limit} Gray/DN, preserve lower-median DN brightness "
            "per LUT; no rescaling after limiting; fractions rounded to 8-bit output"
            if gain_limit else "\nLUT Gain Limit=Off"
        )
        calculated["output_min"] = int(mapped.min())
        calculated["output_max"] = int(mapped.max())
    elif method == "clahe":
        clip_limit = float(params["clahe_clip"])
        grid_size = int(params["clahe_grid"])
        mapped, clahe_info = clahe_unique_levels_to_8bit(
            image, input_max, grid_size, grid_size, clip_limit
        )
        calculated.update(clahe_info)
        calculated.update(
            {
                "ui_clip_limit": clip_limit,
                "clip_normalization": "tile_pixel_count / tile_unique_levels",
                "clahe_grid": grid_size,
            }
        )
        details = (
            f"CLAHE {clahe_info['source_bits']}-bit "
            f"({clahe_info['clahe_implementation']}), "
            f"bins={clahe_info['histogram_bins']}, "
            f"{calculated['input_dn_per_histogram_bin']} DN/bin, "
            f"UI clip={clip_limit:.1f} x tile pixels / tile Unique Levels; "
            f"Unique Levels={clahe_info['tile_unique_levels_min']}.."
            f"{clahe_info['tile_unique_levels_max']} "
            f"(mean {clahe_info['tile_unique_levels_mean']:.1f}); "
            f"internal clip={clahe_info['internal_clip_count_min']}.."
            f"{clahe_info['internal_clip_count_max']} px/bin "
            f"(mean {clahe_info['internal_clip_count_mean']:.1f}), "
            f"grid={grid_size}x{grid_size} -> 8-bit"
        )
    elif method == "fieldscale":
        source = image
        original_shape = source.shape
        padding = (0, 0)
        grid_count = int(params["grid_count"])
        if pad_fieldscale_input:
            padding = ((-source.shape[0]) % grid_count, (-source.shape[1]) % grid_count)
            if any(padding):
                source = np.pad(source, ((0, padding[0]), (0, padding[1])), mode="edge")
        mapped, fieldscale_details = fieldscale_firmware_preview_u8(
            source,
            input_max,
            grid_count=grid_count,
            les_distance=int(params["les_distance"]),
            les_threshold_raw14=int(params["les_threshold_raw14"]),
            message_passing_distance=int(params["message_passing_distance"]),
            message_passing_iterations=int(params["message_passing_iterations"]),
            max_local_gain=float(params.get("max_local_gain", 0.0)),
        )
        if any(padding):
            mapped = mapped[:original_shape[0], :original_shape[1]].copy()
            fieldscale_details.append(
                f"Image edge padding: bottom={padding[0]}, right={padding[1]}; cropped back"
            )
        calculated.update(
            {
                "grid_count": grid_count,
                "les_threshold_percent_raw14": round(
                    int(params["les_threshold_raw14"]) * 100.0 / FIELDSCALE_RAW14_MAX,
                    3,
                ),
            }
        )
        details = " | ".join(fieldscale_details)
    else:
        raise ValueError(f"Unsupported tone mapping method: {method}")

    return np.ascontiguousarray(mapped), details, calculated, params
