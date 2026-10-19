"""features / temporal_drift / analysis."""

from datetime import datetime
from pathlib import Path
import numpy as np
import os
import time
from thermal_toolkit.io.raw import find_raw_files, load_raw


def analyze_temporal_drift(folder, width, height, bit_depth, progress=None, cancelled=None):
    """Read RAW frames once; keep only per-pixel accumulators and frame summaries."""
    folder = str(Path(folder).expanduser().resolve())
    if not Path(folder).is_dir():
        raise ValueError("Select an existing RAW folder.")
    files = find_raw_files(folder)
    if not files:
        raise ValueError("No RAW/BIN files found in the selected folder.")
    total = np.zeros((height, width), dtype=np.float64)
    mean = np.zeros_like(total)
    m2 = np.zeros_like(total)
    minimum = np.full_like(total, np.inf)
    maximum = np.full_like(total, -np.inf)
    frames, skipped = [], []
    first_frame = None
    last_progress = 0.0
    for index, path in enumerate(files, 1):
        if cancelled and cancelled():
            return None
        try:
            before = os.stat(path)
            raw = load_raw(path, width, height)
            after = os.stat(path)
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("File changed while being read; select the folder again when capture is complete.")
        except Exception as exc:
            skipped.append({"filename": Path(path).name, "reason": str(exc)})
        else:
            values = raw.astype(np.float64)
            if first_frame is None:
                first_frame = raw
            count = len(frames) + 1
            total += values
            delta = values - mean
            mean = total / count
            m2 += delta * (values - mean)
            np.minimum(minimum, values, out=minimum)
            np.maximum(maximum, values, out=maximum)
            frames.append({
                "frame_index": count - 1, "filename": Path(path).name,
                "frame_mean_dn": float(values.mean()),
                "frame_spatial_std_dn": float(values.std()),
                "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns,
            })
        now = time.monotonic()
        if progress and (index == len(files) or now - last_progress >= 0.1):
            progress(index, len(files), len(frames), len(skipped))
            last_progress = now
    if cancelled and cancelled():
        return None
    if not frames:
        raise ValueError(f"No valid RAW frames ({len(skipped)} skipped).\n" + skipped[0]["reason"])
    pixel_std = np.sqrt(np.maximum(m2 / len(frames), 0.0))
    pixel_p2p = maximum - minimum
    frame_mean = np.array([row["frame_mean_dn"] for row in frames])

    def distribution(values):
        return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                "max": float(values.max())}

    report = {
        "function": "Temporal Drift Analyzer", "folder": folder,
        "width": width, "height": height, "bit_depth": bit_depth,
        "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_count": len(files), "valid_frames": len(frames), "skipped_files": skipped,
        "frame_order": "Filename sort (same order as frame_statistics.csv); zero-based frame index, not seconds",
        "std_ddof": 0, "units": "DN",
        "definitions": {
            "pixel_temporal_std": "Population standard deviation of each pixel across valid frames",
            "pixel_peak_to_peak": "Maximum minus minimum of each pixel across valid frames",
            "frame_spatial_std": "Population standard deviation across pixels within one frame",
            "p95": "95th percentile across pixels of the selected per-pixel metric",
        },
        "frame_mean": {"first": float(frame_mean[0]), "last": float(frame_mean[-1]),
                       "change": float(frame_mean[-1] - frame_mean[0]),
                       "mean": float(frame_mean.mean()), "std": float(frame_mean.std()),
                       "peak_to_peak": float(np.ptp(frame_mean))},
        "pixel_temporal_std": distribution(pixel_std),
        "pixel_peak_to_peak": distribution(pixel_p2p),
        "notes": ["All statistics use the loaded native DN values; no normalization or detrending.",
                  "Heatmap colors use each map's actual range; compare DN colorbar values across exports."],
    }
    if len(frames) == 1:
        report["notes"].append("Only one valid frame: temporal Std and Peak-to-Peak are zero; temporal variation cannot be assessed.")
    return {"report": report, "frames": frames, "first_frame": first_frame,
            "temporal_mean": mean, "pixel_std": pixel_std, "pixel_peak_to_peak": pixel_p2p}
