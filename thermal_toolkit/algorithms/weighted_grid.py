"""Local histogram-weighted LUTs interpolated at tile centers."""

import numpy as np
from thermal_toolkit.algorithms.histogram import histogram_weighted_mapping_native_u8


def weighted_tiles(image, grid):
    if image.ndim != 2 or not image.size:
        raise ValueError("Weighted mapping requires a non-empty 2D image.")
    if int(grid) != grid or not 1 <= grid <= 64:
        raise ValueError("Weighted mapping grid must be an integer from 1 to 64.")
    grid = int(grid)
    height, width = image.shape
    th, tw = (height + grid - 1) // grid, (width + grid - 1) // grid
    padded = np.pad(image, ((0, th * grid - height), (0, tw * grid - width)), mode="edge")
    for ty in range(grid):
        for tx in range(grid):
            yield ty, tx, padded[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]


def histogram_weighted_grid_u8(image, input_max, clip_limit=2.0, redistribute=True,
                               bright_weight=1.0, bright_percentile=80.0, grid=1,
                               gain_limit=0):
    # Keep the original global path bit-exact, including constant images.
    if int(grid) != grid or not 1 <= grid <= 64:
        raise ValueError("Weighted mapping grid must be an integer from 1 to 64.")
    tiles = weighted_tiles(image, grid)
    if grid == 1:
        next(tiles)  # Validate dimensions and grid even on the global path.
        output, info = histogram_weighted_mapping_native_u8(
            image, input_max, clip_limit, redistribute, bright_weight, bright_percentile,
            gain_limit=gain_limit)
        return output, dict(info, grid=1, interpolation="none")
    grid = int(grid)
    source = np.clip(image, 0, input_max).astype(np.int64)
    height, width = source.shape
    th, tw = (height + grid - 1) // grid, (width + grid - 1) // grid
    luts = np.empty((grid, grid, int(input_max) + 1), dtype=np.uint8)
    records = []
    for ty, tx, tile in tiles:
        data = {}
        _, info = histogram_weighted_mapping_native_u8(
            tile, input_max, clip_limit, redistribute, bright_weight, bright_percentile,
            lut_data=data, gain_limit=gain_limit)
        luts[ty, tx] = data["lut"]
        records.append(dict(info, tile_y=ty, tile_x=tx))

    def neighbors(length, tile_size):
        coordinate = (np.arange(length, dtype=np.float64) + 0.5) / tile_size - 0.5
        lower = np.floor(coordinate).astype(np.int32)
        return np.clip(lower, 0, grid - 1), np.clip(lower + 1, 0, grid - 1), coordinate - lower

    x0, x1, wx = neighbors(width, tw)
    y0, y1, wy = neighbors(height, th)
    output = np.empty(source.shape, dtype=np.uint8)
    for y, row in enumerate(source):
        a = luts[y0[y], x0, row].astype(np.float64)
        b = luts[y0[y], x1, row].astype(np.float64)
        c = luts[y1[y], x0, row].astype(np.float64)
        d = luts[y1[y], x1, row].astype(np.float64)
        top, bottom = a + (b - a) * wx, c + (d - c) * wx
        output[y] = np.clip(np.rint(top + (bottom - top) * wy[y]), 0, 255).astype(np.uint8)
    return output, {
        "grid": grid, "interpolation": "bilinear", "padding": "edge",
        "tile_width": tw, "tile_height": th,
        "padding_right": tw * grid - width, "padding_bottom": th * grid - height,
        "bright_threshold_scope": "per_tile", "tiles": records,
        "gain_limit": records[0]["gain_limit"], "gain_anchor_scope": "per_tile",
    }
