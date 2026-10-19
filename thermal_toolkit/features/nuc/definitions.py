"""features / nuc / definitions."""




NUC_ANALYSIS_HELP = {
    "mean": "平均值：整張 RAW 所有像素的平均亮度，單位 DN（原始數位值，非溫度）。可觀察校正是否改變整體亮度。",
    "std": "空間標準差：整張影像各像素亮度的離散程度，單位 DN，使用母體標準差（ddof=0）。均勻黑體下越小通常代表越均勻；不是時間雜訊。",
    "min": "最小值：整張 RAW 中最低的像素值，單位 DN。容易受單一暗點影響。",
    "max": "最大值：整張 RAW 中最高的像素值，單位 DN。容易受單一亮點影響。",
    "peak_to_peak": "峰對峰值：最大 DN 減最小 DN，表示整張影像的亮度跨度。容易受極端值或壞點影響；一般場景的真實溫差也會增加此值。",
    "median": "中位數：將像素值由小到大排列後的第 50 百分位，單位 DN。相較平均值，較不受少量極端亮暗點影響。",
    "p1_DN": "第 1 百分位：像素值分布中約 1% 位置的 DN，可觀察暗端亮度，比單一最小值穩定。",
    "p99_DN": "第 99 百分位：像素值分布中約 99% 位置的 DN，可觀察亮端亮度，比單一最大值穩定。",
    "unique_levels": "有效灰階數：RAW 中實際出現的不同 DN 值數量。不是位深，也不是像素數；數量多不一定代表校正較好。",
    "zero_pixels": "零值像素數：輸出值剛好等於 0 的像素數。可能是原有零值，也可能由校正後下限裁切產生。",
    "fullscale_pixels": "滿刻度像素數：值剛好等於所選位深最大值的像素數，例如 14-bit 為 16383。可能代表飽和；不包含超出最大值的原始像素。",
    "center_mean_DN": "中心平均值：影像正中央 256×256 區域的平均 DN；影像較小時依實際尺寸縮小區域。",
    "center_std_DN": "中心標準差：中央 256×256 區域內像素的空間標準差，單位 DN；影像較小時縮小區域。用來觀察中心均勻度。",
    "rim_mean_DN": "邊緣平均值：影像四周寬 32 像素邊框內的平均 DN。可與中心平均值比較，觀察周圍偏亮或偏暗。",
    "rim_center_difference_DN": "邊緣與中心差：邊緣平均值減中心平均值，單位 DN。正值表示邊緣較亮，負值表示邊緣較暗。",
    "row_mean_std_DN": "橫列平均值的標準差：先計算每一橫列的平均 DN，再計算這些平均值的標準差。可觀察上下方向不均勻或水平條紋。",
    "column_mean_std_DN": "直欄平均值的標準差：先計算每一直欄的平均 DN，再計算這些平均值的標準差。可觀察左右方向不均勻或垂直條紋。",
    "block64_mean_std_DN": "區塊平均值的標準差：從左上角分成不重疊的 64×64 區塊，計算各區塊平均 DN 的標準差。反映大範圍亮度不均；邊緣不完整區塊不納入，無完整區塊時為 N/A。",
    "block64_mean_range_DN": "區塊平均值極差：最亮區塊的平均 DN 減最暗區塊的平均 DN。使用完整 64×64 區塊，無完整區塊時為 N/A。",
    "local_residual_sigma_DN": "局部殘差穩健標準差：RAW 減去 3×3 中值濾波結果後，排除外圍 2 像素，再以 1.4826×MAD 估計殘差波動，單位 DN。MAD 是殘差相對其中位數之絕對差的中位數。",
    "local_candidate_threshold_DN": "候選異常點門檻：取「8×局部殘差穩健標準差」與「10 DN」的較大者。每張影像各自計算，原始與校正後的門檻可能不同。",
    "local_candidate_pixels": "候選異常像素數：局部殘差與其中位數的差之絕對值超過門檻的像素數，排除外圍 2 像素。真實邊緣也可能被計入，不能直接視為已確認壞點；不同影像的門檻可能不同。",
}


NUC_CHANGE_HELP = {
    "std": "標準差下降量＝原始標準差－校正後標準差（DN）；下降百分比＝下降量÷原始標準差×100%。正值表示下降，負值表示增加；原始值為 0 時百分比為 N/A。",
    "peak_to_peak": "峰對峰下降量＝原始峰對峰值－校正後峰對峰值（DN）；下降百分比＝下降量÷原始峰對峰值×100%。負值表示校正後跨度增加；原始值為 0 時百分比為 N/A。",
    "mean": "平均亮度變化＝校正後平均值－原始平均值（DN）。正值表示變亮，負值表示變暗；方向與下降量的計算不同。",
    "clipped": "下限／上限裁切像素數：整數校正後、裁切前，低於 0 或高於所選位深最大值的像素數。大量裁切也會使標準差下降，應搭配此數值判讀。",
    "pixels": "總像素數＝影像寬度×高度。全圖統計使用所有像素；中心、邊框、區塊及局部殘差使用各自定義的範圍。",
}


NUC_ANALYSIS_FIELDS = (
    ("Mean", "mean", "DN"), ("Spatial Std", "std", "DN"),
    ("Min", "min", "DN"), ("Max", "max", "DN"),
    ("Peak-to-Peak", "peak_to_peak", "DN"), ("Median", "median", "DN"),
    ("P1", "p1_DN", "DN"), ("P99", "p99_DN", "DN"),
    ("Unique levels", "unique_levels", "count"),
    ("Zero pixels", "zero_pixels", "count"), ("Full-scale pixels", "fullscale_pixels", "count"),
    ("Center mean", "center_mean_DN", "DN"), ("Center Std", "center_std_DN", "DN"),
    ("Rim mean", "rim_mean_DN", "DN"), ("Rim - center", "rim_center_difference_DN", "DN"),
    ("Row mean Std", "row_mean_std_DN", "DN"),
    ("Column mean Std", "column_mean_std_DN", "DN"),
    ("64x64 mean Std", "block64_mean_std_DN", "DN"),
    ("64x64 mean range", "block64_mean_range_DN", "DN"),
    ("Residual sigma", "local_residual_sigma_DN", "DN"),
    ("Candidate threshold", "local_candidate_threshold_DN", "DN"),
    ("Candidate pixels", "local_candidate_pixels", "count"),
)


NUC_COMPARISON_ROIS = {
    "方塊": (310, 205, 370, 265),
    "左側參考": (280, 216, 304, 252),
    "右側參考": (376, 216, 400, 252),
}


NUC_ROI_METRICS = ("mean", "std", "min", "max", "median", "peak_to_peak", "p1_DN", "p99_DN", "unique_levels", "zero_pixels", "fullscale_pixels")
