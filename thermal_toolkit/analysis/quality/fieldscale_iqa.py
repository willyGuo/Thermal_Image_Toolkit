"""Static image IQA: explicit image metrics and paired tone-mapping metrics.

Contrast equations/preprocessing follow HENSOLDT ToneMappingIQA (MIT).
See docs/image_quality_metrics.md and tmqi_reference.py for attribution.
"""

import numpy as np
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.algorithms.mapping import raw_to_display_u8

PROFILE = "Fieldscale IQA"
FIELDS = [("Gradient ↑", "gradient"), ("Entropy ↑", "entropy"),
          ("C_global ↓", "c_global"), ("C_local ↓", "c_local"), ("TMQI ↑", "tmqi")]
HELP = {
    "gradient": "8-bit 影像正規化至 0..1 後，3×3 Sobel 梯度幅值的平均（scale=1/8）。越高代表邊緣變化較強，也可能包含雜訊。",
    "entropy": "256-bin Shannon entropy / 8，範圍 0..1。越高代表灰階分布資訊量較多，不保證有效細節更多。",
    "c_global": "全域對比損失，依 RAW 與輸出影像的 log-domain Gaussian 局部標準差計算。越低越好，負值表示對比增加；需要對應原圖。",
    "c_local": "局部對比損失，依 RAW 與輸出影像的 log-domain bilateral residual 計算。越低越好，可為負值；需要對應原圖。",
    "tmqi": "Tone Mapped Image Quality Index，結合五尺度結構保真度及統計自然度。0..1，越高越好。需要對應、非均勻原圖，且寬高至少 161 像素。",
}


def _log_image(image, denominator, gamma):
    image = np.asarray(image)
    normalized = np.power(image / np.float32(denominator), gamma)
    positive_min = np.min(np.where(normalized > 0, normalized, 1))
    normalized = np.where(image <= 0, positive_min, normalized)
    return np.log10(normalized.astype(np.float32))


def _contrast(log_image):
    mean = cv2.GaussianBlur(log_image, (9, 9), 3)
    variance = cv2.GaussianBlur(log_image ** 2, (9, 9), 3) - mean * mean
    global_contrast = -float(np.mean(np.sqrt(np.abs(variance))))
    base = cv2.bilateralFilter(log_image, -1, 0.2, 10)
    local_contrast = float(np.mean(log_image * np.abs(log_image - base)))
    return global_contrast, local_contrast


def image_quality_analysis(image, input_max, reference=None, reference_max=None):
    """No implicit A/B pairing: reference must be explicitly supplied by caller."""
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("Image quality requires a finite, non-empty grayscale image.")
    if input_max <= 0:
        raise ValueError("Image range must be positive.")
    display = raw_to_display_u8(source, input_max)
    normalized = display.astype(np.float64) / 255.0
    gx = cv2.Sobel(normalized, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(normalized, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8)
    gradient_map = np.hypot(gx, gy)
    counts = np.bincount(display.ravel(), minlength=256)
    probabilities = counts[counts > 0] / display.size
    metrics = {"quality_profile": PROFILE, "gradient": float(gradient_map.mean()),
               "entropy": float(-np.sum(probabilities * np.log2(probabilities)) / 8),
               "c_global": None, "c_local": None, "tmqi": None}
    reasons = {key: "需要對應的原始影像；兩張獨立輸入不自動視為 RAW／輸出配對。"
               for key in ("c_global", "c_local", "tmqi")}
    if reference is not None:
        reference = np.asarray(reference)
        if reference.shape != source.shape or not np.isfinite(reference).all():
            raise ValueError("Reference must be finite and have the same dimensions.")
        if reference_max is None or reference_max <= 0:
            raise ValueError("Reference range must be positive.")
        ref = np.clip(reference, 0, reference_max)
        hdr_log = _log_image(ref, reference_max + 1, 1.0)
        ldr_log = _log_image(display, 256, 2.2)
        hdr_global, hdr_local = _contrast(hdr_log)
        ldr_global, ldr_local = _contrast(ldr_log)
        metrics.update(c_global=ldr_global - hdr_global, c_local=ldr_local - hdr_local)
        reasons.clear()
        if min(source.shape) < 161:
            reasons["tmqi"] = "五尺度 TMQI 需要寬高至少 161 像素；不自動放大或減少尺度。"
        elif np.ptp(ref) == 0:
            reasons["tmqi"] = "原始影像為常數，TMQI 的 HDR 動態範圍正規化無法定義。"
        else:
            from thermal_toolkit.analysis.quality.tmqi_reference import TMQI
            evaluator = TMQI()
            hdr, ldr = evaluator.img_preprocessing(ref.astype(np.float64), display.astype(np.float64))
            with np.errstate(invalid="ignore", divide="ignore"):
                structure = evaluator.StructuralFidelity(hdr, ldr, evaluator.level, evaluator.weight)
                naturalness = evaluator.StatisticalNaturalness(ldr)
                value = evaluator.a * structure ** evaluator.Alpha + (1 - evaluator.a) * naturalness ** evaluator.Beta
            if np.isfinite(value):
                metrics["tmqi"] = float(value)
            else:
                reasons["tmqi"] = "此影像的 TMQI 計算不產生有限實數。"
    return {"metrics": metrics, "unavailable_reasons": reasons,
            "display_u8": display,
            "gradient_u8": np.clip(np.rint(gradient_map * 255), 0, 255).astype(np.uint8)}


def quality_text(result):
    return ", ".join(f"{key}=" + ("N/A" if result['metrics'][key] is None
                      else f"{result['metrics'][key]:.5f}") for _, key in FIELDS)
