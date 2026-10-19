"""reports / pipeline."""

import numpy as np
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.io.export import save_png


def save_pipeline_overview_png(path, items, max_columns=3):
    """Save numbered pipeline images as one labeled contact sheet."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required to save PNG. Install: pip install opencv-python")
    if not items:
        raise ValueError("Pipeline overview requires at least one image.")

    prepared = []
    for label, image, display_max in items:
        if image.ndim != 2:
            raise ValueError("Pipeline overview supports grayscale images only.")
        if display_max <= 0:
            raise ValueError("Pipeline overview display maximum must be positive.")
        image_u8 = np.rint(
            np.clip(image, 0, display_max).astype(np.float64)
            * 255.0
            / display_max
        ).astype(np.uint8)
        prepared.append((str(label), image_u8))

    source_height, source_width = prepared[0][1].shape
    if any(image.shape != (source_height, source_width) for _, image in prepared):
        raise ValueError("All pipeline overview images must have the same dimensions.")

    scale = min(1.0, 480.0 / source_width, 360.0 / source_height)
    tile_width = max(1, int(round(source_width * scale)))
    tile_height = max(1, int(round(source_height * scale)))
    header_height = 42
    gap = 8
    margin = 8
    columns = min(max(1, int(max_columns)), len(prepared))
    rows = (len(prepared) + columns - 1) // columns
    panel_width = tile_width
    panel_height = header_height + tile_height
    canvas_width = margin * 2 + columns * panel_width + (columns - 1) * gap
    canvas_height = margin * 2 + rows * panel_height + (rows - 1) * gap
    canvas = np.full((canvas_height, canvas_width), 24, dtype=np.uint8)

    for item_index, (label, image_u8) in enumerate(prepared):
        row = item_index // columns
        column = item_index % columns
        x = margin + column * (panel_width + gap)
        y = margin + row * (panel_height + gap)
        resized = (
            image_u8
            if image_u8.shape == (tile_height, tile_width)
            else cv2.resize(
                image_u8,
                (tile_width, tile_height),
                interpolation=cv2.INTER_AREA,
            )
        )
        canvas[y + header_height:y + panel_height, x:x + tile_width] = resized
        cv2.rectangle(
            canvas,
            (x, y),
            (x + panel_width - 1, y + panel_height - 1),
            72,
            1,
        )
        font_scale = min(
            0.62,
            max(0.36, panel_width / max(500.0, len(label) * 17.0)),
        )
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        text_x = x + max(6, (panel_width - text_size[0]) // 2)
        cv2.putText(
            canvas,
            label,
            (text_x, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            255,
            1,
            cv2.LINE_AA,
        )

    save_png(path, canvas)


def save_uvc_comparison_png(path, original, processed, algorithm_title):
    if cv2 is None:
        raise RuntimeError("OpenCV is required to save PNG. Install: pip install opencv-python")
    if original.ndim != 2 or processed.ndim != 2:
        raise ValueError("UVC comparison PNG requires two grayscale images.")
    if original.shape != processed.shape:
        raise ValueError(
            f"UVC comparison image size mismatch: original={original.shape}, processed={processed.shape}"
        )

    original_u8 = np.ascontiguousarray(np.clip(original, 0, 255).astype(np.uint8))
    processed_u8 = np.ascontiguousarray(np.clip(processed, 0, 255).astype(np.uint8))
    height, width = original_u8.shape
    header_height = 42
    gap = 8
    canvas = np.full((height + header_height, width * 2 + gap), 24, dtype=np.uint8)
    canvas[header_height:, :width] = original_u8
    canvas[header_height:, width + gap:] = processed_u8

    labels = ("Original UVC", f"Processed: {algorithm_title}")
    panel_x = (0, width + gap)
    for label, x_offset in zip(labels, panel_x):
        font_scale = min(0.7, max(0.42, width / max(420.0, len(label) * 18.0)))
        text_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        text_x = x_offset + max(6, (width - text_size[0]) // 2)
        cv2.putText(
            canvas,
            label,
            (text_x, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            255,
            1,
            cv2.LINE_AA,
        )

    save_png(path, canvas)
