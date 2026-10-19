"""features / high_low / page."""

import csv
import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8, signed_delta_to_display_u8
from thermal_toolkit.io.export import save_png
from thermal_toolkit.io.raw import load_raw
from thermal_toolkit.ui.widgets.inputs import FocusWheelSpinBox, PathRow


class HighLowPageMixin:
    """features / high_low / page behavior composed by the main window."""

    def build_high_low_page(self):
        # High Low Analyzer
        p, f = self._form_page(); self.an_low = PathRow("file", "RAW Files (*.raw *.RAW *.bin *.BIN)", "Select LOW RAW File for High / Low Analysis"); self.an_high = PathRow("file", "RAW Files (*.raw *.RAW *.bin *.BIN)", "Select HIGH RAW File for High / Low Analysis"); self.an_min_delta = FocusWheelSpinBox(); self.an_min_delta.setRange(0, 16383); self.an_min_delta.setValue(50); f.addRow("LOW RAW", self.an_low); f.addRow("HIGH RAW", self.an_high); f.addRow("Min Delta DN", self.an_min_delta); self.stack.addWidget(p)

    def run_high_low(self):
        low = load_raw(self.an_low.text(), self.width, self.height); high = load_raw(self.an_high.text(), self.width, self.height)
        diff = high.astype(np.int32) - low.astype(np.int32); threshold = self.an_min_delta.value()
        valid = diff > threshold
        out = self.make_output("high_low_analysis")
        np.save(out / "delta.npy", diff)
        yy, xx = np.where(~valid)
        with open(out / "abnormal_pixels.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["x","y","low","high","delta","high_le_low","small_delta","saturated"])
            for y,x in zip(yy,xx):
                d=int(diff[y,x]); wr.writerow([x,y,int(low[y,x]),int(high[y,x]),d,int(d<=0),int(0<d<=threshold),int(low[y,x]>=self.raw_max or high[y,x]>=self.raw_max)])
        # Signed delta uses the fixed Bit Depth range; zero maps to middle gray.
        delta8 = signed_delta_to_display_u8(diff, self.raw_max)
        save_png(out / "difference_preview.png", delta8)
        self.current_original = low; self.current_result = diff
        self.preview_original.set_array(raw_to_display_u8(low, self.raw_max)); self.preview_result.set_array(delta8)
        self.log_stats("LOW", low); self.log_stats("HIGH", high)
        self.log.append(f"High > Low: {(diff>0).sum():,} ({(diff>0).mean()*100:.3f}%)")
        self.log.append(f"High = Low: {(diff==0).sum():,}; High < Low: {(diff<0).sum():,}")
        self.log.append(f"Delta Mean={diff.mean():.3f}, Std={diff.std():.3f}, Min={diff.min()}, Max={diff.max()}")
        self.log.append(f"Delta > {threshold} valid: {valid.sum():,}/{valid.size:,}")
        self.log.append(f"Difference preview scale: fixed {-self.raw_max}..{self.raw_max} DN")
