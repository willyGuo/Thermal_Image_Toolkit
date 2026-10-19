"""algorithms / pipeline."""

import numpy as np
import time
from thermal_toolkit.algorithms.enhancement import apply_image_algorithm
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.algorithms.parameters import algorithm_display_name, default_algorithm_parameters
from thermal_toolkit.core.opencv import cv2


def apply_algorithm_pipeline(
    gray,
    stages,
    input_max=255,
    clahe_cache=None,
    fieldscale_states=None,
    fieldscale_temporal=False,
    allow_empty=False,
    pad_fieldscale_input=False,
):
    """Apply enabled stages in top-to-bottom order and keep every step result."""
    if cv2 is None:
        raise RuntimeError("OpenCV not installed. pip install opencv-python")
    if gray.ndim != 2:
        raise ValueError("Algorithm pipeline input must be a 2D grayscale image.")
    if not any(stage.enabled for stage in stages):
        if allow_empty:
            return raw_to_display_u8(gray, input_max), []
        raise ValueError("Add and enable at least one algorithm stage.")

    processed = np.ascontiguousarray(gray.copy())
    current_max = int(input_max)
    records = []
    execution_index = 0
    for pipeline_index, stage in enumerate(stages, start=1):
        if not stage.enabled:
            continue
        execution_index += 1
        params = default_algorithm_parameters(stage.algorithm_key)
        params.update(stage.parameters)
        clahe = None
        if stage.algorithm_key == "clahe" and clahe_cache is not None:
            cache_key = (
                pipeline_index,
                float(params["clahe_clip"]),
                int(params["clahe_grid"]),
            )
            clahe = clahe_cache.get(cache_key)
            if clahe is None:
                clahe = cv2.createCLAHE(
                    clipLimit=cache_key[1],
                    tileGridSize=(cache_key[2], cache_key[2]),
                )
                clahe_cache[cache_key] = clahe

        fieldscale_state = None
        if stage.algorithm_key == "fieldscale" and fieldscale_states is not None:
            fieldscale_state = fieldscale_states.setdefault(pipeline_index, {})

        started = time.perf_counter()
        original_shape = processed.shape
        padding = (0, 0)
        if pad_fieldscale_input and stage.algorithm_key == "fieldscale":
            grid = int(params.get("grid_count", 8))
            if grid >= 2:
                padding = ((-original_shape[0]) % grid, (-original_shape[1]) % grid)
                if any(padding):
                    processed = np.pad(processed, ((0, padding[0]), (0, padding[1])), mode="edge")
        processed, details = apply_image_algorithm(
            processed,
            stage.algorithm_key,
            low_percentile=params.get("low_percentile", 1.0),
            high_percentile=params.get("high_percentile", 99.0),
            percentile_gain_limit=params.get("percentile_gain_limit", 0.0),
            clahe_clip=params.get("clahe_clip", 2.0),
            clahe_grid=params.get("clahe_grid", 8),
            gamma=params.get("gamma", 1.5),
            median_kernel_size=params.get("median_kernel_size", 3),
            gaussian_kernel_size=params.get("gaussian_kernel_size", 5),
            gaussian_sigma=params.get("gaussian_sigma", 1.0),
            bilateral_strength=params.get("bilateral_strength", 2),
            unsharp_amount=params.get("unsharp_amount", 1.0),
            clahe=clahe,
            input_max=current_max,
            fieldscale_state=fieldscale_state,
            fieldscale_temporal=(
                fieldscale_temporal and stage.algorithm_key == "fieldscale"
            ),
            grid_count=params.get("grid_count", 8),
            les_distance=params.get("les_distance", 2),
            les_threshold_raw14=params.get("les_threshold_raw14", 400),
            message_passing_distance=params.get(
                "message_passing_distance", 1
            ),
            message_passing_iterations=params.get(
                "message_passing_iterations", 7
            ),
            temporal_smoothing_percent=params.get(
                "temporal_smoothing_percent", 90
            ),
            max_local_gain=params.get("max_local_gain", 0.0),
            dde_sigma_s=params.get("dde_sigma_s", 21),
            dde_sigma_r=params.get("dde_sigma_r", 51),
            dde_gain_percent=params.get("dde_gain_percent", 200),
            dde_num_iters=params.get("dde_num_iters", 3),
            histogram_weighted_grid=params.get("histogram_weighted_grid", 1),
            histogram_weighted_clip=params.get("histogram_weighted_clip", 2.0),
            histogram_weighted_redistribute=params.get("histogram_weighted_redistribute", True),
            histogram_weighted_bright_weight=params.get("histogram_weighted_bright_weight", 1.0),
            histogram_weighted_bright_percentile=params.get("histogram_weighted_bright_percentile", 80.0),
            histogram_weighted_gain_limit=params.get("histogram_weighted_gain_limit", 0),
        )
        if any(padding):
            processed = processed[:original_shape[0], :original_shape[1]].copy()
            details.append(
                f"Image edge padding: bottom={padding[0]}, right={padding[1]}; cropped back to original size"
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        current_max = 255
        records.append(
            {
                "pipeline_index": pipeline_index,
                "execution_index": execution_index,
                "algorithm": stage.algorithm_key,
                "name": algorithm_display_name(stage.algorithm_key),
                "parameters": dict(params),
                "details": list(details),
                "elapsed_ms": elapsed_ms,
                "image": processed,
            }
        )
    return processed, records


def pipeline_execution_json(records):
    return [
        {
            "pipeline_index": record["pipeline_index"],
            "execution_index": record["execution_index"],
            "algorithm": record["algorithm"],
            "name": record["name"],
            "parameters": dict(record["parameters"]),
            "details": list(record["details"]),
            "elapsed_ms": round(float(record["elapsed_ms"]), 3),
        }
        for record in records
    ]


def pipeline_summary_text(stages):
    names = [algorithm_display_name(stage.algorithm_key) for stage in stages if stage.enabled]
    return " -> ".join(names) if names else "No enabled algorithm"


def pipeline_record_image(records, pipeline_index, final_image):
    if pipeline_index is None:
        return final_image
    for record in records:
        if record["pipeline_index"] == pipeline_index:
            return record["image"]
    return final_image
