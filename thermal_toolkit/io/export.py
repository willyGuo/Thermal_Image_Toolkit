"""io / export."""

import numpy as np
from thermal_toolkit.core.opencv import cv2


def save_csv_matrix(path, data, fmt):
    np.savetxt(path, data, delimiter=",", fmt=fmt)


def save_png(path, image):
    if cv2 is None:
        raise RuntimeError("OpenCV is required to save PNG. Install: pip install opencv-python")
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def save_rgb_png(path, image_rgb):
    if cv2 is None:
        raise RuntimeError("OpenCV is required to save PNG. Install: pip install opencv-python")
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("RGB PNG output requires a three-channel image.")
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb[:, :, :3]), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")
