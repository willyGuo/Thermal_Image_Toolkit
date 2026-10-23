"""algorithms / parameters."""

from dataclasses import dataclass
from dataclasses import field


BPR_MARK_RADIUS = 5


BPR_HIGHLIGHT_RADIUS = 2


IMAGE_ALGORITHM_OPTIONS = [
    ("Percentile Auto Contrast", "auto_contrast"),
    ("Fieldscale Local Adaptive AGC (Firmware Core)", "fieldscale"),
    ("Histogram Equalization", "hist_equalization"),
    ("CLAHE", "clahe"),
    ("Gamma", "gamma"),
    ("Bilateral Denoise", "bilateral"),
    ("Unsharp Mask", "unsharp"),
    ("DDE Domain Transform Detail Enhancement", "dde"),
    ("Auto Contrast + CLAHE", "auto_contrast_clahe"),
    ("P1/P99 Percentile Preview (Original Values)", "raw_percentile_preview"),
]


PIPELINE_ALGORITHM_OPTIONS = [
    ("Percentile Auto Contrast", "auto_contrast"),
    ("Fieldscale Local Adaptive AGC (Firmware Core)", "fieldscale"),
    ("Histogram Equalization", "hist_equalization"),
    ("Histogram-weighted mapping", "histogram_weighted"),
    ("CLAHE", "clahe"),
    ("Gamma", "gamma"),
    ("Median Filter", "median"),
    ("Gaussian Filter", "gaussian"),
    ("Bilateral Denoise", "bilateral"),
    ("Unsharp Mask", "unsharp"),
    ("DDE Domain Transform Detail Enhancement", "dde"),
]


TONE_MAPPING_OPTIONS = [
    ("Linear Full Range", "linear"),
    ("Percentile", "percentile"),
    ("Histogram-weighted mapping", "histogram_weighted"),
    ("Effective-range Weighted / Double Plateau", "weighted_plateau"),
    ("CLAHE Tone Mapping", "clahe"),
    ("Fieldscale Local Adaptive AGC", "fieldscale"),
]


DEFAULT_TONE_MAPPING_PARAMETERS = {
    "weighted_plateau": {
        "plateau_low_percentile": 0.0,
        "plateau_high_percentile": 100.0,
        "plateau_mode": "double_plateau",
        "plateau_upper": 1.0,
        "plateau_lower": 0.1,
        "plateau_redistribute": False,
        "plateau_gain_limit": 0.0,
    },
    "linear": {"black_dn": 0, "white_dn": 16383},
    "percentile": {"low_percentile": 1.0, "high_percentile": 99.0, "percentile_gain_limit": 0.0},
    "histogram_weighted": {
        "histogram_weighted_grid": 1,
        "histogram_weighted_clip": 2.0,
        "histogram_weighted_redistribute": True,
        "histogram_weighted_bright_weight": 1.0,
        "histogram_weighted_bright_percentile": 80.0,
        "histogram_weighted_gain_limit": 0,
    },
    "clahe": {"clahe_clip": 2.0, "clahe_grid": 8},
    "fieldscale": {
        "grid_count": 8,
        "les_distance": 2,
        "les_threshold_raw14": 400,
        "message_passing_distance": 1,
        "message_passing_iterations": 7,
        "max_local_gain": 0.0,
    },
}


IMAGE_ALGORITHM_DESCRIPTIONS = {
    "histogram_weighted": (
        "Histogram-weighted mapping on the preceding step's 8-bit output (0..255). "
        "Uses adjacent-bin weights, Unique-Level clipping, full-input-range redistribution, "
        "bright weighting and optional final LUT Gain Limit. Grid 1 is global; "
        "larger grids interpolate per-tile LUTs. Gain Limit is per 8-bit input level, not RAW DN."
    ),
    "auto_contrast": (
        "在 8-bit 工作影像上計算 P1/P99，將 P1 映射為黑色、P99 映射為白色，"
        "範圍外像素裁切；不包含 CLAHE。"
    ),
    "fieldscale": (
        "Firmware-matching Fieldscale core: builds 8x8 local Min/Max fields, "
        "applies configurable LES and message passing, and locally maps the input "
        "range to the full display range. Gamma and CLAHE are not applied. RAW "
        "Compare uses original DN; UVC Compare also provides Temporal Smoothing."
    ),
    "hist_equalization": (
        "對整張 8-bit 影像執行全域直方圖均衡化，重新分配灰階以提升整體對比。"
    ),
    "clahe": (
        "將 8-bit 影像分區做限制對比的局部直方圖均衡化；Clip Limit 限制放大量，"
        "Grid Size 控制分區數量。"
    ),
    "gamma": (
        "以 Gamma 曲線重新映射 8-bit 灰階；目前公式中 Gamma 小於 1 會提亮中間灰階，"
        "大於 1 會壓暗。"
    ),
    "median": (
        "Median filtering replaces each 8-bit pixel with the neighborhood median. "
        "It is especially useful for isolated hot/cold impulse noise while preserving edges."
    ),
    "gaussian": (
        "Gaussian filtering smooths the 8-bit image with a weighted neighborhood. "
        "Kernel Size controls the area and Sigma controls the blur strength."
    ),
    "bilateral": (
        "在 8-bit 影像上做保邊雙邊濾波，降低雜訊並盡量保留物體邊緣。"
    ),
    "unsharp": (
        "以原圖減去模糊影像取得高頻細節，再依 Amount 加回原圖以增強邊緣。"
    ),
    "dde": (
        "Y8 Domain Transform filtering from Tiffany_thermal/Enhancement/DDE. "
        "Output = base + gain * (original - base), clipped to 0..255. "
        "Sigma S controls spatial smoothing; Sigma R controls edge sensitivity. "
        "Gain 100% preserves the input; Standard/Fast use 3/1 iterations."
    ),
    "auto_contrast_clahe": (
        "先在 8-bit 工作影像做 P1/P99 Auto Contrast，再執行 CLAHE 局部對比增強。"
    ),
    "raw_percentile_preview": (
        "直接在輸入的原始數值上計算 P1/P99。P1 是排除最低 1% 像素後的灰階下限，"
        "P99 是排除最高 1% 像素前的灰階上限；將 P1 映射為黑色 0、P99 映射為白色 255，"
        "範圍外的像素裁切，最後輸出成 8-bit 顯示影像。"
    ),
}


DEFAULT_ALGORITHM_PARAMETERS = {
    "histogram_weighted": dict(DEFAULT_TONE_MAPPING_PARAMETERS["histogram_weighted"]),
    "auto_contrast": dict(DEFAULT_TONE_MAPPING_PARAMETERS["percentile"]),
    "fieldscale": {
        "grid_count": 8,
        "les_distance": 2,
        "les_threshold_raw14": 400,
        "message_passing_distance": 1,
        "message_passing_iterations": 7,
        "max_local_gain": 0.0,
    },
    "hist_equalization": {},
    "clahe": {"clahe_clip": 2.0, "clahe_grid": 8},
    "gamma": {"gamma": 1.5},
    "median": {"median_kernel_size": 3},
    "gaussian": {"gaussian_kernel_size": 5, "gaussian_sigma": 1.0},
    "bilateral": {"bilateral_strength": 2},
    "unsharp": {"unsharp_amount": 1.0},
    "dde": {"dde_sigma_s": 21, "dde_sigma_r": 51, "dde_gain_percent": 200, "dde_num_iters": 3},
}


def algorithm_display_name(algorithm_key):
    for display_name, key in PIPELINE_ALGORITHM_OPTIONS:
        if key == algorithm_key:
            return display_name
    return algorithm_key


def default_algorithm_parameters(algorithm_key):
    if algorithm_key not in DEFAULT_ALGORITHM_PARAMETERS:
        raise ValueError(f"Unsupported pipeline algorithm: {algorithm_key}")
    return dict(DEFAULT_ALGORITHM_PARAMETERS[algorithm_key])


@dataclass
class AlgorithmStage:
    algorithm_key: str
    enabled: bool = True
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        defaults = default_algorithm_parameters(self.algorithm_key)
        defaults.update(self.parameters)
        if self.algorithm_key == "dde":
            # Gaussian-pyramid parameters have no Domain Transform equivalent.
            for legacy_key in ("dde_sigma", "dde_alpha", "dde_beta"):
                defaults.pop(legacy_key, None)
        self.parameters = defaults

    def clone(self):
        return AlgorithmStage(self.algorithm_key, self.enabled, dict(self.parameters))

    def to_dict(self):
        return {
            "algorithm": self.algorithm_key,
            "name": algorithm_display_name(self.algorithm_key),
            "enabled": bool(self.enabled),
            "parameters": dict(self.parameters),
        }
