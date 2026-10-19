"""algorithms / fieldscale."""

import numpy as np


FIELDSCALE_RAW14_MAX = 16383


FIELDSCALE_Q8_ONE = 256


FIELDSCALE_INTERP_ONE = 65536


def _fieldscale_suppress_extrema(
    grid, threshold_q8, suppress_maximum, distance=2
):
    result = grid.copy()
    height, width = grid.shape
    distance = max(1, int(distance))
    for y in range(height):
        for x in range(width):
            neighbors = []
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbors.append(int(grid[ny, nx]))
            if not neighbors:
                continue
            current = int(grid[y, x])
            mean_value = sum(neighbors) // len(neighbors)
            if suppress_maximum:
                if (
                    current >= max(neighbors)
                    and current > mean_value
                    and current - mean_value > threshold_q8
                ):
                    result[y, x] = mean_value + threshold_q8
            elif (
                current <= min(neighbors)
                and mean_value > current
                and mean_value - current > threshold_q8
            ):
                result[y, x] = mean_value - threshold_q8
    return result


def _fieldscale_message_pass(grid, increase, distance=1):
    result = grid.copy()
    height, width = grid.shape
    distance = max(1, int(distance))
    for y in range(height):
        for x in range(width):
            values = []
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        values.append(int(grid[ny, nx]))
            current = int(grid[y, x])
            mean_value = sum(values) // len(values)
            result[y, x] = (
                max(mean_value, current) if increase else min(mean_value, current)
            )
    return result


def _fieldscale_resize_q8(grid, output_width, output_height):
    grid_height, grid_width = grid.shape
    x_start_q16 = grid_width * 32768 // output_width - 32768
    x_step_q16 = grid_width * FIELDSCALE_INTERP_ONE // output_width
    y_start_q16 = grid_height * 32768 // output_height - 32768
    y_step_q16 = grid_height * FIELDSCALE_INTERP_ONE // output_height
    source_x = x_start_q16 + np.arange(output_width, dtype=np.int64) * x_step_q16
    source_y = y_start_q16 + np.arange(output_height, dtype=np.int64) * y_step_q16
    source_x = np.clip(source_x, 0, (grid_width - 1) * FIELDSCALE_INTERP_ONE)
    source_y = np.clip(source_y, 0, (grid_height - 1) * FIELDSCALE_INTERP_ONE)
    x0 = (source_x >> 16).astype(np.intp)
    y0 = (source_y >> 16).astype(np.intp)
    x1 = np.minimum(x0 + 1, grid_width - 1)
    y1 = np.minimum(y0 + 1, grid_height - 1)
    fx = source_x & 0xFFFF
    fy = source_y & 0xFFFF

    horizontal = (
        grid[:, x0] * (FIELDSCALE_INTERP_ONE - fx)[None, :]
        + grid[:, x1] * fx[None, :]
        + FIELDSCALE_INTERP_ONE // 2
    ) // FIELDSCALE_INTERP_ONE
    return (
        horizontal[y0, :] * (FIELDSCALE_INTERP_ONE - fy)[:, None]
        + horizontal[y1, :] * fy[:, None]
        + FIELDSCALE_INTERP_ONE // 2
    ) // FIELDSCALE_INTERP_ONE


def fieldscale_firmware_preview_u8(
    image,
    input_max,
    state=None,
    temporal=False,
    grid_count=8,
    les_distance=2,
    les_threshold_raw14=400,
    message_passing_distance=1,
    message_passing_iterations=7,
    temporal_smoothing_percent=90,
    max_local_gain=0.0,
):
    if image.ndim != 2:
        raise ValueError("Fieldscale input must be a 2D grayscale image.")
    height, width = image.shape
    grid_count = int(grid_count)
    if (
        grid_count < 1
        or width < grid_count
        or height < grid_count
        or width % grid_count
        or height % grid_count
    ):
        raise ValueError(
            "Fieldscale requires width and height divisible by the selected "
            f"Grid count ({grid_count}x{grid_count})."
        )
    if input_max <= 0:
        raise ValueError("Fieldscale input maximum must be positive.")

    source = np.clip(image, 0, input_max).astype(np.int64)
    patch_height = height // grid_count
    patch_width = width // grid_count
    patches = source.reshape(
        grid_count,
        patch_height,
        grid_count,
        patch_width,
    )
    min_grid = patches.min(axis=(1, 3)) * FIELDSCALE_Q8_ONE
    max_grid = patches.max(axis=(1, 3)) * FIELDSCALE_Q8_ONE
    global_grid = grid_count == 1
    les_distance = (
        0 if global_grid else max(1, min(grid_count - 1, int(les_distance)))
    )
    message_passing_distance = (
        0
        if global_grid
        else max(1, min(grid_count - 1, int(message_passing_distance)))
    )
    message_passing_iterations = max(0, int(message_passing_iterations))
    if global_grid:
        message_passing_iterations = 0
    les_threshold_raw14 = max(0, int(les_threshold_raw14))
    temporal_smoothing_percent = max(
        0, min(99, int(temporal_smoothing_percent))
    )
    max_local_gain = max(0.0, float(max_local_gain))
    threshold_q8 = max(
        0,
        int(round(
            les_threshold_raw14 * input_max * FIELDSCALE_Q8_ONE
            / FIELDSCALE_RAW14_MAX
        )),
    )
    if not global_grid:
        max_grid = _fieldscale_suppress_extrema(
            max_grid, threshold_q8, True, les_distance
        )
        max_grid = _fieldscale_suppress_extrema(
            max_grid, threshold_q8, False, les_distance
        )
        for _ in range(message_passing_iterations):
            min_grid = _fieldscale_message_pass(
                min_grid, False, message_passing_distance
            )
            max_grid = _fieldscale_message_pass(
                max_grid, True, message_passing_distance
            )

    temporal_applied = False
    if temporal and state is not None:
        previous_min = state.get("min_grid")
        previous_max = state.get("max_grid")
        previous_input_max = state.get("input_max")
        if (
            previous_min is not None
            and previous_max is not None
            and previous_min.shape == min_grid.shape
            and previous_max.shape == max_grid.shape
            and previous_input_max == input_max
        ):
            previous_weight = temporal_smoothing_percent
            current_weight = 100 - previous_weight
            min_grid = (
                current_weight * min_grid
                + previous_weight * previous_min
                + 50
            ) // 100
            max_grid = (
                current_weight * max_grid
                + previous_weight * previous_max
                + 50
            ) // 100
            temporal_applied = True
        state["min_grid"] = min_grid.copy()
        state["max_grid"] = max_grid.copy()
        state["input_max"] = input_max

    gain_limit_applied = max_local_gain >= 1.0
    minimum_range_levels = 0.0
    if gain_limit_applied:
        full_range_q8 = int(input_max) * FIELDSCALE_Q8_ONE
        minimum_range_q8 = min(
            full_range_q8,
            int(np.ceil(full_range_q8 / max_local_gain)),
        )
        low_grid = np.minimum(min_grid, max_grid)
        high_grid = np.maximum(min_grid, max_grid)
        span_grid = np.maximum(high_grid - low_grid, minimum_range_q8)
        adjusted_low = (low_grid + high_grid - span_grid) // 2
        adjusted_low = np.maximum(adjusted_low, 0)
        adjusted_low = np.minimum(adjusted_low, full_range_q8 - span_grid)
        min_grid = adjusted_low
        max_grid = adjusted_low + span_grid
        minimum_range_levels = minimum_range_q8 / FIELDSCALE_Q8_ONE

    min_field = _fieldscale_resize_q8(min_grid, width, height)
    max_field = _fieldscale_resize_q8(max_grid, width, height)
    low = np.minimum(min_field, max_field)
    high = np.maximum(min_field, max_field)
    value_q8 = source * FIELDSCALE_Q8_ONE
    output = np.zeros(source.shape, dtype=np.int64)
    output[value_q8 >= high] = input_max
    valid = (value_q8 > low) & (value_q8 < high) & (high > low)
    output[valid] = (
        (value_q8[valid] - low[valid]) * input_max
        // (high[valid] - low[valid])
    )
    preview = np.rint(output.astype(np.float64) * 255.0 / input_max).astype(np.uint8)
    threshold_levels = threshold_q8 / FIELDSCALE_Q8_ONE
    details = [
        (
            "Fieldscale global Min/Max mode: grid=1x1; LES=off"
            if global_grid
            else f"Fieldscale firmware core: grid={grid_count}x{grid_count}, "
                 f"LES distance={les_distance}"
        ),
        f"LES threshold={les_threshold_raw14} RAW14 DN "
        f"({threshold_levels:.2f} input levels)",
        (
            "message passing=off (global 1x1)"
            if global_grid
            else f"message passing distance={message_passing_distance}, "
                 f"iterations={message_passing_iterations}"
        ),
        (
            f"max local gain={max_local_gain:.1f}x; "
            f"minimum local range={minimum_range_levels:.1f} input levels"
            if gain_limit_applied
            else "max local gain=off (reference behavior)"
        ),
        "Gamma=off, CLAHE=off",
    ]
    if temporal:
        details.append(
            f"temporal=current {100 - temporal_smoothing_percent}% / "
            f"previous {temporal_smoothing_percent}%"
            if temporal_applied
            else "temporal=first frame"
        )
    return preview, details
