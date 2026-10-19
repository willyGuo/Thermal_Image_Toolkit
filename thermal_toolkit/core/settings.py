"""core / settings."""

import json
import numpy as np
from thermal_toolkit.core.constants import (
    ACTIVE_SPATIAL_QUALITY_WEIGHTS,
    ACTIVE_UVC_TEMPORAL_WEIGHTS,
    BIT_DEPTH_OPTIONS,
    DEFAULT_BIT_DEPTH,
    DEFAULT_HEIGHT,
    DEFAULT_SPATIAL_QUALITY_WEIGHTS,
    DEFAULT_UVC_TEMPORAL_WEIGHTS,
    DEFAULT_WIDTH,
    QUALITY_TIMELINE_METRICS,
    RESOLUTION_OPTIONS,
)
from thermal_toolkit.core.paths import default_output_root, toolkit_settings_path
from thermal_toolkit.algorithms.weighted_aci import default_weighted_aci_settings, validate_weighted_aci_settings


def default_toolkit_settings():
    return {
        "bad_pixel_calibration": {"dc_sigma": 5., "response_percent": 5., "netd_sigma": 5., "window": 9},
        "weighted_mapping": default_weighted_aci_settings(),
        "log_font_size": 17,
        "raw_input": {
            "resolution": [DEFAULT_WIDTH, DEFAULT_HEIGHT],
            "bit_depth": DEFAULT_BIT_DEPTH,
            "output_root": default_output_root(),
        },
        "timeline": {
            "metric": "overall_score",
            "history_seconds": 60,
            "raw_histogram_show_cdf": False,
        },
        "camera_mirror": {
            "camera_a": True,
            "camera_b": False,
        },
        "quality_weights": {
            "spatial": dict(DEFAULT_SPATIAL_QUALITY_WEIGHTS),
            "uvc_temporal": dict(DEFAULT_UVC_TEMPORAL_WEIGHTS),
        },
    }


def validated_toolkit_settings(settings):
    defaults = default_toolkit_settings()
    if not isinstance(settings, dict):
        return defaults
    result = default_toolkit_settings()
    bad_pixel = settings.get('bad_pixel_calibration', {})
    if not isinstance(bad_pixel, dict):
        bad_pixel = {}
    for key, default in defaults['bad_pixel_calibration'].items():
        try:
            value = float(bad_pixel.get(key, default))
            if not np.isfinite(value):
                raise ValueError()
            if key == 'window':
                value = int(np.clip(value, 3, 31))
                if value % 2 == 0:
                    value += 1
            else:
                value = float(np.clip(value, .01, 1000 if key == 'response_percent' else 100))
        except (TypeError, ValueError, OverflowError):
            value = default
        result['bad_pixel_calibration'][key] = value
    result["weighted_mapping"] = validate_weighted_aci_settings(settings.get("weighted_mapping", {}))
    result["log_font_size"] = int(
        np.clip(settings.get("log_font_size", 17), 10, 28)
    )
    raw_input = settings.get("raw_input", {})
    try:
        resolution = tuple(
            int(value)
            for value in raw_input.get(
                "resolution", (DEFAULT_WIDTH, DEFAULT_HEIGHT)
            )
        )
    except (TypeError, ValueError):
        resolution = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    if resolution not in RESOLUTION_OPTIONS:
        resolution = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    try:
        bit_depth = int(raw_input.get("bit_depth", DEFAULT_BIT_DEPTH))
    except (TypeError, ValueError):
        bit_depth = DEFAULT_BIT_DEPTH
    if bit_depth not in BIT_DEPTH_OPTIONS:
        bit_depth = DEFAULT_BIT_DEPTH
    output_root = str(raw_input.get("output_root", "")).strip()
    if not output_root:
        output_root = default_output_root()
    result["raw_input"] = {
        "resolution": list(resolution),
        "bit_depth": bit_depth,
        "output_root": output_root,
    }
    timeline = settings.get("timeline", {})
    valid_metrics = {metric for _label, metric in QUALITY_TIMELINE_METRICS}
    metric = timeline.get("metric", "overall_score")
    result["timeline"]["metric"] = (
        metric if metric in valid_metrics else "overall_score"
    )
    result["timeline"]["history_seconds"] = int(
        np.clip(timeline.get("history_seconds", 60), 10, 600)
    )
    result["timeline"]["raw_histogram_show_cdf"] = bool(timeline.get("raw_histogram_show_cdf", False))
    camera_mirror = settings.get("camera_mirror", {})
    if not isinstance(camera_mirror, dict):
        camera_mirror = {}
    result["camera_mirror"] = {
        "camera_a": bool(camera_mirror.get("camera_a", True)),
        "camera_b": bool(camera_mirror.get("camera_b", False)),
    }
    source_weights = settings.get("quality_weights", {})
    for section, defaults_for_section in (
        ("spatial", DEFAULT_SPATIAL_QUALITY_WEIGHTS),
        ("uvc_temporal", DEFAULT_UVC_TEMPORAL_WEIGHTS),
    ):
        candidate = source_weights.get(section, {})
        values = {
            key: int(np.clip(candidate.get(key, value), 0, 100))
            for key, value in defaults_for_section.items()
        }
        if sum(values.values()) != 100:
            values = dict(defaults_for_section)
        result["quality_weights"][section] = values
    return result


def load_toolkit_settings():
    path = toolkit_settings_path()
    if not path.is_file():
        return default_toolkit_settings()
    try:
        return validated_toolkit_settings(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, TypeError):
        return default_toolkit_settings()


def save_toolkit_settings(settings):
    path = toolkit_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".new")
    temporary_path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def apply_quality_weights(settings):
    weights = settings["quality_weights"]
    ACTIVE_SPATIAL_QUALITY_WEIGHTS.clear()
    ACTIVE_SPATIAL_QUALITY_WEIGHTS.update(weights["spatial"])
    ACTIVE_UVC_TEMPORAL_WEIGHTS.clear()
    ACTIVE_UVC_TEMPORAL_WEIGHTS.update(weights["uvc_temporal"])
