"""features / bpr / render."""

import numpy as np
from thermal_toolkit.algorithms.parameters import BPR_HIGHLIGHT_RADIUS, BPR_MARK_RADIUS
from thermal_toolkit.core.opencv import cv2


def draw_bpr_markers(
    base_gray,
    coordinates,
    radius=BPR_MARK_RADIUS,
    highlight_bright_points=False,
    highlight_radius=BPR_HIGHLIGHT_RADIUS,
):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for BPR marking. Install: pip install opencv-python")
    if base_gray.ndim != 2:
        raise ValueError("BPR marker base image must be a 2D grayscale image.")
    marked_rgb = cv2.cvtColor(
        np.ascontiguousarray(np.clip(base_gray, 0, 255).astype(np.uint8)),
        cv2.COLOR_GRAY2RGB,
    )
    for x, y in coordinates:
        cv2.circle(
            marked_rgb,
            (int(x), int(y)),
            int(radius),
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if highlight_bright_points:
            cv2.circle(
                marked_rgb,
                (int(x), int(y)),
                int(highlight_radius),
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
    return marked_rgb
