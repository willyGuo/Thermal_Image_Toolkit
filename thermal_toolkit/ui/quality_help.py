"""ui / quality_help."""

import os
from thermal_toolkit.core.paths import bundled_resource_path
from thermal_toolkit.core.qt import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QUrl,
    QVBoxLayout,
    Qt,
)


class QualityScoreHelpDialog(QDialog):
    def __init__(self, include_temporal=False, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        source_name = "UVC" if include_temporal else "RAW"
        self.setWindowTitle(f"{source_name} Quality Score Guide")
        self.setModal(False)
        self.resize(820, 720)
        self.setMinimumSize(620, 480)
        self.include_temporal = bool(include_temporal)

        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.reload_guide()
        layout.addWidget(self.browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def reload_guide(self):
        guide_path = bundled_resource_path(
            "html/quality_score_guide.html" if self.include_temporal else "html/image_quality_guide.html")
        if guide_path.is_file():
            self.browser.document().setBaseUrl(
                QUrl.fromLocalFile(f"{guide_path.parent}{os.sep}")
            )
            self.browser.setHtml(guide_path.read_text(encoding="utf-8"))
            self.browser.scrollToAnchor(
                "uvc" if self.include_temporal else "raw"
            )
        else:
            self.browser.setHtml(self._help_html(self.include_temporal))

    @staticmethod
    def _help_html(include_temporal):
        if not include_temporal:
            from thermal_toolkit.analysis.quality.fieldscale_iqa import FIELDS, HELP
            return "<h2>Image Quality Metrics</h2>" + "".join(
                f"<p><b>{name}</b>：{HELP[key]}</p>" for name, key in FIELDS)
        profile_title = "UVC Lightweight Quality" if include_temporal else "Image Quality"
        profile_description = (
            "UVC 即時評分採降採樣與單尺度 Gaussian/Sobel 分析，避免品質監測影響影像輸出；"
            "另以約 10 Hz 取樣計算 Flicker 與 Temporal Stability。"
            if include_temporal
            else
            "RAW 靜態評分使用原始位深與完整解析度。Flat 區先依灰階分成 8 個 level，"
            "再依連通區域拆開，因此天空、牆面等不同灰階或不同位置的平坦區不會直接混成同一組統計。"
        )
        total_formula = (
            "0.70 × Spatial Score + 0.15 × Flicker Control + 0.15 × Temporal Stability"
            if include_temporal
            else
            "0.22 × Detail Clarity + 0.18 × Contrast Quality + 0.18 × Flat Cleanliness + "
            "0.16 × Thermal Uniformity + 0.10 × Clipping Control + 0.16 × Artifact Control"
        )
        temporal_rows = ""
        if include_temporal:
            temporal_rows = """
            <tr><td><b>閃爍控制<br>Flicker Control</b></td>
                <td>去除亮度線性趨勢後的 frame brightness 波動。</td>
                <td><code>100 × exp(-flicker_DN / 2)</code></td>
                <td>越高代表可見亮度閃爍越少。</td></tr>
            <tr><td><b>時間穩定度<br>Temporal Stability</b></td>
                <td>穩定平坦區內，去除全域 offset 後的 frame-to-frame robust residual noise。</td>
                <td><code>100 × exp(-temporal_noise_DN / 3)</code></td>
                <td>越高代表連續影像越穩定。</td></tr>
            """

        raw_extra = "" if include_temporal else """
          <h2>RAW Flat 區域如何拆分</h2>
          <p>1. Bilateral Filter 建立 Base Layer，Detail = Original - Base。</p>
          <p>2. Sobel gradient + 9×9 Local Std 找出 Flat candidate，並避開 Edge 周圍。</p>
          <p>3. Flat pixel 依 Base gray level 的 P1~P99 範圍切成 8 個灰階帶。</p>
          <p>4. 每個灰階帶再用 8-connectivity Connected Components 分成獨立區域。</p>
          <p>5. 每個區域各自算 Robust Noise / Uniformity；Typical 使用面積加權 P50，Worst 使用各有效區域的 P80（不按面積加權）。
             Score 不是只看最大面積區域，因此大面積乾淨天空不會完全掩蓋較小但較差的牆面。</p>
          <p><b>Quality Confidence</b> 只表示本張影像有多少 Flat / Edge 可供分析，不直接加入 Total Score。</p>
        """

        return f"""
        <html><head><style>
          body {{ color:#e6edf2; background:#202328; font-family:'Segoe UI','Microsoft JhengHei'; font-size:14px; line-height:1.55; }}
          h1 {{ color:#86c8f4; font-size:24px; margin-bottom:6px; }}
          h2 {{ color:#c9e2f2; font-size:18px; margin-top:20px; }}
          table {{ border-collapse:collapse; width:100%; margin-top:10px; }}
          th {{ background:#31546c; color:#ffffff; padding:8px; text-align:left; }}
          td {{ border:1px solid #53606c; padding:8px; vertical-align:top; }}
          code {{ color:#ffd38a; font-family:'Cascadia Mono','Consolas'; }}
          .notice {{ background:#2b323a; border-left:4px solid #d6a85f; padding:10px; }}
        </style></head><body>
          <h1>{profile_title}</h1>
          <p class="notice"><b>判讀原則：</b>0~100 分，越高越好。這仍是 No-Reference、同場景相對比較指標，
          不是絕對 PASS/FAIL 規格；沒有 Ground Truth 時，任何單一總分都不能完全代表主觀畫質。</p>
          <p>{profile_description}</p>

          <h2>Total Score</h2><p><code>{total_formula}</code></p>
          {"<p>Spatial Score 使用相同六個空間維度，但 UVC 使用 Lightweight 計算。時序資料至少累積 1.5 秒才顯示完整 Total Score。</p>" if include_temporal else ""}

          <h2>主要分數</h2>
          <table>
            <tr><th>分數</th><th>評估內容</th><th>核心方法</th><th>高分代表</th></tr>
            <tr><td><b>細節清晰度<br>Detail Clarity</b></td>
                <td>真正邊緣強度與 Detail SNR，同時扣除 Halo。</td>
                <td>Sobel Edge + Base/Detail + Robust Flat Noise。RAW：<code>50% Sharpness + 50% Detail-SNR - 8% Halo Penalty</code></td>
                <td>細節強、相對 noise 高，而且不是單純 oversharpen。</td></tr>
            <tr><td><b>對比品質<br>Contrast Quality</b></td>
                <td>Robust P1~P99 動態範圍 + Histogram Entropy。</td>
                <td>RAW：70% Span + 30% Entropy；UVC：75% Span + 25% Entropy。</td>
                <td>灰階利用充分，且不是只靠少數極端值拉大範圍。</td></tr>
            <tr><td><b>平坦區潔淨度<br>Flat Cleanliness</b></td>
                <td>Flat 區的高頻 residual noise。</td>
                <td>RAW 對每個灰階/連通區域分開算 MAD robust sigma；Typical 使用面積加權 P50，Worst 使用各有效區域 P80（不按面積加權）。另外以 3×3 Median residual + 小型 Connected Component 抓 isolated hot/cold impulse。</td>
                <td>不同亮度平坦區都較乾淨，且少量異常 pixel 不會被 robust MAD 完全忽略。</td></tr>
            <tr><td><b>熱影像均勻度<br>Thermal Uniformity</b></td>
                <td>低頻不均、局部 residual，以及 RAW 額外的 row/column banding。</td>
                <td>RAW：50% Typical + 30% Worst + 20% Banding Control；UVC 使用簡化 residual。</td>
                <td>平坦熱區更均勻，條紋/帶狀 fixed-pattern 較少。</td></tr>
            <tr><td><b>裁切控制<br>Clipping Control</b></td>
                <td>靠近黑端/白端的 clipping pixel 比例。</td>
                <td><code>100 × exp(-clipping_percent / 3)</code></td>
                <td>黑白細節較少因 AGC/contrast 拉伸而消失。</td></tr>
            <tr><td><b>假影控制<br>Artifact Control</b></td>
                <td>Edge Halo + 水平/垂直 Seam/Block discontinuity；重複 seam 比單一自然邊界懲罰更重。</td>
                <td>50% Halo Control + 50% Seam Control；RAW Halo 以 edge-normal overshoot/undershoot 判斷，Seam 主要抓重複且近似週期性的 block/grid line。</td>
                <td>較少 halo、ringing、block/grid boundary。</td></tr>
            {temporal_rows}
          </table>
          {raw_extra}
        </body></html>
        """
