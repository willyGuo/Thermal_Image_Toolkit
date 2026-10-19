"""io / raw."""

from pathlib import Path
import glob
import numpy as np
import os
from thermal_toolkit.core.constants import RAW_DTYPE
from thermal_toolkit.core.opencv import cv2


def load_raw(path, width, height):
    expected_pixels = width * height
    expected_bytes = expected_pixels * 2
    size = os.path.getsize(path)
    if size != expected_bytes:
        raise ValueError(
            f"RAW size error:\n{path}\nActual: {size:,} bytes\nExpected: {expected_bytes:,} bytes "
            f"({width}x{height}x2)"
        )
    raw = np.fromfile(path, dtype=RAW_DTYPE)
    return raw.reshape((height, width))


IMAGE_INPUT_FILTER = "RAW / Images (*.raw *.RAW *.bin *.BIN *.jpg *.JPG *.jpeg *.JPEG *.png *.PNG *.bmp *.BMP *.tif *.TIF *.tiff *.TIFF)"


IMAGE_INPUT_NOTE = ("JPG / PNG / BMP / TIFF: automatic dimensions and bit depth; color is converted to grayscale. "
                    "RAW resolution / bit-depth settings apply only to RAW / BIN files.")


IMAGE_INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_analysis_image(path, width, height, bit_depth):
    """Decode image files to native-depth grayscale; preserve RAW loading rules."""
    is_image = Path(path).suffix.lower() in IMAGE_INPUT_EXTENSIONS
    if is_image:
        pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.size == 0:
            raise ValueError(f"Cannot decode image: {path}")
        if pixels.dtype not in (np.uint8, np.uint16):
            raise ValueError("Only unsigned 8-bit and 16-bit images are supported.")
        if pixels.ndim == 3:
            channels = pixels.shape[2]
            if channels not in (3, 4):
                raise ValueError("Unsupported image channel count.")
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY)
        if pixels.ndim != 2:
            raise ValueError("Expected a single grayscale image.")
        bit_depth = pixels.dtype.itemsize * 8
    else:
        pixels = load_raw(path, width, height)
    return pixels, {
        "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
        "bit_depth": int(bit_depth), "value_max": (1 << bit_depth) - 1,
        "value_name": "Gray level" if is_image else "RAW DN",
        "source_type": "image" if is_image else "raw",
    }


def find_raw_files(folder):
    files = []
    for ext in ("*.raw", "*.RAW", "*.bin", "*.BIN"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def average_raw_folder(folder, width, height):
    files = find_raw_files(folder)
    if not files:
        raise FileNotFoundError(f"No RAW/BIN files found:\n{folder}")
    total = np.zeros((height, width), dtype=np.float64)
    valid = []
    skipped = []
    for path in files:
        try:
            total += load_raw(path, width, height)
            valid.append(path)
        except Exception as exc:
            skipped.append((path, str(exc)))
    if not valid:
        raise ValueError("No valid RAW files in selected folder.")
    avg = np.rint(total / len(valid)).astype(np.uint16)
    return avg, valid, skipped
