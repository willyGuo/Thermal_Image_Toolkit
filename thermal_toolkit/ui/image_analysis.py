"""ui / image_analysis."""

from thermal_toolkit.io.raw import load_analysis_image
from thermal_toolkit.reports.histogram import render_gray_histogram_rgb


class ImageAnalysisMixin:
    """ui / image_analysis behavior composed by the main window."""

    def load_analysis_source(self, path):
        pixels, info = load_analysis_image(path, self.width, self.height, self.bit_depth)
        self.log.append(
            f"[{info['source_type'].title()}] Loaded: {path}\n"
            f"{info['width']}x{info['height']}, "
            f"{info['bit_depth']}-bit {info['value_name']} (0..{info['value_max']}). "
            + ("Image dimensions/depth are automatic; RAW settings are ignored. Color is converted to grayscale." if info['source_type'] == "image" else "")
        )
        return pixels, info

    def set_image_save_ready(self, mode, ready):
        setattr(self, f"_{mode}_image_save_ready", bool(ready))
        name = "Single Image Statistics" if mode == "single" else "Dual Image Compare"
        if self.FUNCTIONS[self.current_function_index()] == name:
            self.run_btn.setEnabled(bool(ready))

    def refresh_raw_histogram_cdf(self):
        show_cdf = self.app_settings["timeline"].get("raw_histogram_show_cdf", False)
        for prefix in ("raw_viewer", "raw_algorithm"):
            group = getattr(self, prefix + "_histogram_group", None)
            if group is not None:
                group.setTitle("RAW DN Histogram" + (" + CDF" if show_cdf else ""))
            stats = getattr(self, prefix + "_stats", None)
            if stats is None:
                continue
            info = (self.raw_algorithm_statistics_info if prefix == "raw_algorithm"
                    else self.raw_viewer_info)
            if prefix == "raw_algorithm":
                group.setTitle(info["value_name"] + " Histogram" + (" + CDF" if show_cdf else ""))
            rgb = render_gray_histogram_rgb(
                getattr(self, prefix + "_histogram"), stats,
                value_max=info["value_max"], show_cdf=show_cdf,
                title=f"{info['bit_depth']}-bit {info['value_name']} Histogram (256 display bins)",
            )
            setattr(self, prefix + "_histogram_rgb", rgb)
            getattr(self, prefix + "_histogram_preview").set_array(rgb)
        window = self.raw_viewer_histogram_window
        if window is not None and window.isVisible():
            if self.FUNCTIONS[self.current_function_index()] == "Image Algorithm Compare":
                self.open_algorithm_histogram_window()
            else:
                self.open_raw_histogram_window()

    @staticmethod
    def populate_raw_stat_labels(values, stats):
        values["count"].setText(f"{stats['count']:,}")
        values["unique_levels"].setText(
            f"{stats['unique_levels']:,}"
        )
        values["min"].setText(str(stats["min"]))
        values["max"].setText(str(stats["max"]))
        for key in ("mean", "median", "std", "p1", "p5", "p95", "p99"):
            values[key].setText(f"{stats[key]:,.3f}")
        values["entropy_bits"].setText(
            f"{stats['entropy_bits']:.3f} bits"
        )
        values["p99_p1"].setText(
            f"{stats['p99_p1']:,.3f}"
        )
        values["p99_p1_percent"].setText(
            f"{stats['p99_p1_percent']:.3f}%"
        )
        values["zero_clip"].setText(
            f"{stats['zero_count']:,} ({stats['zero_percent']:.3f}%)"
        )
        values["max_clip"].setText(
            f"{stats['max_count']:,} ({stats['max_percent']:.3f}%)"
        )
        values["over_max"].setText(
            f"{stats['over_max_count']:,} "
            f"({stats['over_max_percent']:.3f}%)"
        )

    @staticmethod
    def write_raw_viewer_analysis_summary(summary, stats, value_name="RAW DN"):
        summary.write(f"{value_name} statistics:\n")
        summary.write(
            f"  Pixels / unique DN: {stats['count']:,} / "
            f"{stats['unique_levels']:,}\n"
        )
        summary.write(
            f"  Min / max / bit max: {stats['min']} / "
            f"{stats['max']} / {stats['raw_max']}\n"
        )
        summary.write(
            f"  Mean / median / std: {stats['mean']:.3f} / "
            f"{stats['median']:.3f} / {stats['std']:.3f}\n"
        )
        summary.write(
            f"  P1 / P5 / P95 / P99: {stats['p1']:.3f} / "
            f"{stats['p5']:.3f} / {stats['p95']:.3f} / "
            f"{stats['p99']:.3f}\n"
        )
        summary.write(
            f"  P99-P1: {stats['p99_p1']:.3f} "
            f"({stats['p99_p1_percent']:.3f}% of bit range)\n"
        )
        summary.write(
            f"  Entropy: {stats['entropy_bits']:.3f} bits\n"
        )
        summary.write(
            f"  Zero clip: {stats['zero_count']:,} "
            f"({stats['zero_percent']:.3f}%)\n"
        )
        summary.write(
            f"  Bit max clip: {stats['max_count']:,} "
            f"({stats['max_percent']:.3f}%)\n"
        )
        summary.write(
            f"  Over bit max: {stats['over_max_count']:,} "
            f"({stats['over_max_percent']:.3f}%)\n"
        )
