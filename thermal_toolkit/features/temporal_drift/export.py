"""features / temporal_drift / export."""

from pathlib import Path
import csv
import json
import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.core.constants import RAW_DTYPE
from thermal_toolkit.io.export import save_png


def save_temporal_drift_charts(out, result):
    """Export labeled, noninteractive figures without changing the Qt backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    report = result["report"]
    subtitle = (f"{report['valid_frames']} frames | {report['width']} x {report['height']} | "
                f"{report['bit_depth']}-bit input | population Std (ddof=0)")
    paths = []

    def figure(title):
        fig = Figure(figsize=(10, 7), dpi=160, layout="constrained", facecolor="white")
        FigureCanvasAgg(fig)
        fig.suptitle(title + "\n" + subtitle, fontsize=13)
        return fig

    def save(fig, name):
        fig.savefig(Path(out) / name, dpi=160)
        fig.clear()
        paths.append(name)

    fig = figure("Frame brightness and spatial variation")
    axes = fig.subplots(2, 1, sharex=True)
    indices = [row["frame_index"] for row in result["frames"]]
    for ax, key, label, color in zip(
        axes, ("frame_mean_dn", "frame_spatial_std_dn"),
        ("Frame mean (DN)", "Frame spatial Std (DN)"), ("#2369a2", "#bd5b18")
    ):
        ax.plot(indices, [row[key] for row in result["frames"]], color=color,
                linewidth=1.5, marker="." if len(indices) < 100 else None)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("Frame index (filename order; not elapsed time)")
    save(fig, "frame_statistics_plot.png")

    fig = figure("Distribution across pixels")
    axes = fig.subplots(2, 1)
    for ax, key, summary_key, title, color in zip(
        axes, ("pixel_std", "pixel_peak_to_peak"),
        ("pixel_temporal_std", "pixel_peak_to_peak"),
        ("Pixel temporal Std", "Pixel Peak-to-Peak"), ("#2369a2", "#bd5b18")
    ):
        values = result[key].ravel()
        stats = report[summary_key]
        ax.hist(values, bins=80, range=(0, max(float(values.max()), 1.0)), color=color, alpha=0.8)
        ax.axvline(stats["p95"], color="#9c2551", linestyle="--", label=f"P95 = {stats['p95']:.3f} DN")
        ax.set_title(f"{title}: mean {stats['mean']:.3f}, max {stats['max']:.3f} DN", fontsize=11)
        ax.set_xlabel(title + " (DN)")
        ax.set_ylabel("Pixel count")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    save(fig, "pixel_distributions.png")

    for key, title, name in (
        ("pixel_std", "Pixel temporal Std", "temporal_std_heatmap.png"),
        ("pixel_peak_to_peak", "Pixel Peak-to-Peak", "peak_to_peak_heatmap.png"),
    ):
        values = result[key]
        low, high = float(values.min()), float(values.max())
        # A constant map still needs a nonzero, explicitly labeled color range.
        display_high = high if high > low else low + 1.0
        fig = figure(title + f" | actual range {low:.3f} .. {high:.3f} DN")
        ax = fig.subplots()
        plotted = ax.imshow(values, origin="upper", cmap="viridis", interpolation="nearest",
                            vmin=low, vmax=display_high)
        ax.set_xlabel("X (pixel)")
        ax.set_ylabel("Y (pixel)")
        fig.colorbar(plotted, ax=ax, label=title + " (DN)")
        save(fig, name)
    return paths


def save_temporal_drift_results(out, result):
    """Save the cached analysis shown in the UI, without reading the input again."""
    out = Path(out)
    report = dict(result["report"])
    raw_max = (1 << report["bit_depth"]) - 1
    result["temporal_mean"].astype(RAW_DTYPE).tofile(out / "temporal_mean.raw")
    for key in ("pixel_std", "pixel_peak_to_peak"):
        np.save(out / (key + ".npy"), result[key])
    for key, name in (("temporal_mean", "temporal_mean_preview.png"),
                      ("pixel_std", "temporal_std_preview.png"),
                      ("pixel_peak_to_peak", "peak_to_peak_preview.png")):
        save_png(out / name, raw_to_display_u8(result[key], raw_max))
    with open(out / "frame_statistics.csv", "w", newline="", encoding="utf-8-sig") as handle:
        fields = ("frame_index", "filename", "frame_mean_dn", "frame_spatial_std_dn")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["frames"])
    report["charts"] = save_temporal_drift_charts(out, result)
    report["frames"] = result["frames"]
    with open(out / "temporal_drift_summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report
