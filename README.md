# Thermal Image Toolkit

PyQt5 熱影像分析工具，支援 RAW／一般影像、NUC、BPR、Temporal Drift，以及 Image／UVC 演算法比較。

## 啟動

```powershell
python -m pip install PyQt5 numpy opencv-python openpyxl matplotlib scipy
python main.py
```

也可以使用 `python -m thermal_toolkit`。`main.py` 是啟動與相容入口，實作已移至 `thermal_toolkit/`；執行或交付原始碼時需保留整個套件資料夾。

## 程式結構

`Calibration > Bad pixel calibration` uses DC Level, Responsivity and optional NETD.
DC-only needs RAW A; B and temperatures are not required. Responsivity needs A/B;
NETD additionally needs temperatures and a temporal folder. Unused inputs are disabled.
Input A/B: uniform blackbody captures at different temperatures, headerless little-endian
uint16, without BPR/NUC/AGC. Resolution follows Setting; enter both temperatures.
DC compares A against the valid surrounding 9x9 neighbours (centre excluded):
absolute median residual > 5 times population local standard deviation.
Responsivity compares B-A against each pixel's surrounding 9x9 response median
(centre excluded, valid border neighbours): relative deviation > 5%. The 9x9 window
is fixed independently of the DC window. Equality is not flagged; if the local reference
is zero, the DN-domain threshold is zero and percentages are N/A.
NETD uses a same-temperature RAW/BIN folder (100 frames recommended, minimum 2):
population temporal std / absolute two-temperature slope, reported in mK.
Absolute deviation from full-frame valid NETD mean > 5 times its population std is flagged;
zero/opposite response slopes are flagged separately by NETD. Thresholds and DC window
are editable. These slide criteria identify candidates, not proof of permanent failure.
Run exports the union of enabled masks, per-point method labels and measurements in
`bad_pixel_coordinates.xlsx` (DC Level / Responsivity / NETD worksheets), parameters/source list/counts in `calibration_summary.json`,
and a 16,384-byte Novatek SIE `dpc_table.bin` (maximum 4,095 points; no truncation).
Both input RAWs and all temporal input frames are preserved in the result folder.
Invalid inputs are rejected before export. Above 4,095 candidates, preview, all input RAWs,
Excel and statistics are still saved; only dpc_table.bin is skipped. Summary JSON/TXT
records the skip reason, per-method counts, union percentage and method overlaps.

Percentile Tone Mapping 與 Image／UVC 的 `Percentile Auto Contrast` Enhancement 提供
`Gain Limit (Gray / DN)`：0.0～4.0、步進 0.1，0.0 為 Off。當原本 `255 / (High DN - Low DN)`
超過上限，降低線性拉伸斜率，保留中位 DN 原本的亮度與百分位範圍外的裁切，不再拉滿 0～255。
小數限制在 8-bit 取整數前生效；Tone Mapping 使用原始 DN，Enhancement 使用前一步的 8-bit 灰階。
參數 `percentile_gain_limit` 會隨 Save 寫入 `pipeline.json`，Reset 恢復 Off。

靜態影像的 Image Quality 使用 Gradient、Entropy、C_global、C_local、TMQI，
取代舊的加權總分；UVC 已移除評分，保留預覽、擷取、錄影與一般統計。參考影像需求與公式見
[品質指標說明](html/image_quality_guide.html)。打包時需包含整個 `html/` 資料夾。

Image Algorithm Compare 的 Histogram-weighted mapping 提供 `Weighted Mapping Grid Size`：
`1` 保留全圖映射，`8` 代表 8×8 格。Grid 大於 1 時，各格獨立計算 clipping、亮部百分位與權重，
再以雙線性插值混合相鄰四格的 LUT。無法整除的尺寸以邊緣像素補齊，輸出維持原圖大小。
Save 的 `histogram_weighted_intervals.csv` 在分區模式包含從 0 起算的 `tile_y`、`tile_x`；
比率代表各格插值前的映射，`pipeline.json` 同時記錄 Grid、插值方式及各格計算資訊。

`Gain Limit (Gray / DN)` 可輸入 0.0～4.0，精度／步進為 0.1，0.0（預設）表示 Off。
在原本 Weighted Mapping 完成後，限制 LUT 每增加 1 DN 的灰階增量。以原始像素分布的中位 DN 原本輸出亮度為基準
（偶數像素取排序後較低的中位值），限制後不再拉滿 0～255；空影像／單一 DN 沿用原本的零輸出。
Grid 大於 1 時，每格獨立限制再插值；此限制不保證跨格像素亮度差，也不限制後續 Enhancement。
變更立即更新預覽，Reset 恢復 Off；Save 的 `pipeline.json` 記錄 `histogram_weighted_gain_limit`，
CSV 保留原有權重比例，新增 `before_gain_output_span`、`final_output_span` 與區間起訖灰階，供限制前後比較。
第一輪黑體實驗可用 Grid 1，關閉後續 Enhancement，再比較 Off／1／2／4。
小數增量先以十分之一灰階累加，最後四捨五入為 8-bit 整數輸出；因此 0.5 的限制在取整數後
仍可能出現 1 Gray 的跳動。原本 Off／1／2／4 的影像結果不變。
CSV 的 `limited_output_span_before_rounding` 是取整數前的增量，`final_output_span` 則是實際整數 LUT 差。

Image Algorithm Compare 的 8-bit Enhancement 也可加入 `Histogram-weighted mapping`，
使用前一步的 0～255 影像，沿用 Grid、Unique-Level Clip、Redistribution、Bright Weight 與 Gain Limit。
預設 Grid 1、Clip 2、Redistribution 開啟、Bright Weight 1（Off）、P80、Gain Limit Off；
此階段的 Gain Limit 單位是每 1 階 8-bit 輸入灰階，不是原始 RAW DN。
共用 Pipeline 編輯器的 UVC 同步提供此選項；各步參數、執行說明與結果影像由原有 Save 流程保存。

```text
main.py                      程式啟動入口與 API 匯出
thermal_toolkit/
  app.py                    建立 QApplication 與主視窗
  core/                     版本、共用常數、設定與資源路徑
  io/                       RAW／一般影像讀取、共用檔案輸出
  algorithms/               Tone Mapping、CLAHE、Fieldscale、DDE、Pipeline
  analysis/                 影像統計、空間與 UVC 時間品質分析
  reports/                  Histogram、品質 Excel、比較圖片等共用匯出
  features/                 各功能頁面的介面與操作流程
    average_raw/            Average RAW Folder
    nuc/                    NUC Calculate、NUC_Analyze
    high_low/               High / Low Analyzer
    temporal_drift/         Temporal Drift Analyzer
    bpr/                    BPR Bad Pixel Marking
    image_statistics/       Single Image Statistics
    dual_image/             Dual Image Compare
    image_algorithm/        Image Algorithm Compare
    single_uvc/             Single UVC Statistic
    dual_uvc/               Dual UVC Compare
    uvc_algorithm/          UVC Algorithm Compare
  ui/                       主視窗、共用控制項、視窗、樣式與設定介面
  uvc/                      相機列舉、擷取、處理、錄影、影格緩衝
tests/                      演算法、模組整合及離屏 GUI 測試
html/                       HTML 說明文件（quality_score_guide.html）
docs/architecture.md        模組責任、依賴方向與維護指引
```

各功能通常由 `page.py` 負責建立頁面及處理操作；較大的功能另分 `analysis.py`、`export.py`、`worker.py` 或 `recording.py`。詳細對照見 [架構說明](docs/architecture.md)。

`docs/toolkit_settings.json`、`html/quality_score_guide.html` 和預設的 `Thermal_RAW_Results/` 以原始碼根目錄為基準；打包後仍使用 EXE 所在目錄及封裝資源位置，說明文件保留 `html/` 子目錄。

## 驗證

```powershell
python -m unittest discover -s tests -v
```

測試使用暫存目錄及離屏 Qt，不會開啟實體相機。演算法快照涵蓋重構前的 18 組 Tone Mapping／Enhancement 輸出；若日後刻意修改演算法，需同步審查對應測試基準。

目前工作目錄未保留 PyInstaller spec；日後打包以 `main.py` 為入口，並在 spec 的 `datas` 加入 `('html', 'html')`。這次結構整理沒有變更版本號或產生 EXE；實體 UVC 與打包版本需另外驗證。


## Weighted mapping Automatic Clip

`Setting > Weighted mapping` provides automatic Clip and four editable thresholds:
Range THL/THH = 75/350 and Standard deviation THL/THH = 18/121.
Range is P99 minus P1; standard deviation uses all input pixels (`ddof=0`).
Each statistic is normalized between its own THL/THH and clamped to 0..1.
Ratio Combination selects MIN (default), Average, or MAX of the two normalized ratios.
The selected ratio controls `Clip = 20 + ratio * (2 * pixels / UniqueLevels - 20)`.
For small tiles whose Clip max is below 20, the lower endpoint is bounded by Clip max.
The resulting px/bin cap is rounded to an integer (minimum 1).

Automatic Clip is enabled by default in the application and overrides the manual
Clip field, without changing its saved value. Disable it to restore manual clipping.
Redistribution, Bright Weight and the existing final-LUT Gain Limit remain separate controls.
Grid > 1 measures each padded tile independently. Thresholds are in input DN units;
8-bit Enhancement uses its preceding 0..255 image, so adjust thresholds accordingly.
Apply updates Image previews and the running UVC pipeline, and saves the settings in
`docs/toolkit_settings.json` under `weighted_mapping`. THL must be smaller than THH.
Reset to Default restores the four thresholds; Apply commits the reset.
Explicit Save includes `weighted_mapping_settings` in Image/UVC `pipeline.json`;
interval CSV calculations use the same active automatic Clip settings.


Weighted mapping redistribution now covers the full input DN range, including
originally empty levels, while preserving the per-bin cap. The redistributed LUT
uses every interval from 0 through input_max; without redistribution it keeps the
populated-range mapping. With 640x512 RAW14, Automatic Clip = 20 px/bin, Bright
Weight = 1 and final LUT Gain Limit Off, redistribution fills all 16384 bins with
20 counts and produces `round(DN * 255 / 16383)`. Automatic redistribution also
handles constant frames through this full-range mapping.

`weighted_mapping.ratio_mode` persists the Ratio Combination as `min`, `mean`, or `max`.
Older settings without this key use MIN. Apply refreshes the calculated explanation
and processing; Reset restores MIN. The final LUT Gain Limit remains independent.

Weighted mapping also supports `Parameters` in Settings: two parameters (the existing
Range + Standard deviation, default), or one parameter `(P99 - P1) / mean`.
Single mode uses the mean of all pixels in each tile, with independent dimensionless
THL/THH (defaults 0/1), and `ratio = clamp((statistic - THL) / (THH - THL), 0, 1)`.
A zero-mean tile uses statistic 0. The existing Clip interpolation is unchanged.
Ratio Combination applies only to two-parameter mode. Switching modes retains both
sets of thresholds; Apply persists `parameter_mode` (`dual`/`single`) and
`relative_range_thl`/`relative_range_thh` with the shared Image/UVC settings.
Older settings without these fields continue to use two parameters.
