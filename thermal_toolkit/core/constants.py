"""core / constants."""

from thermal_toolkit.core.qt import Qt


DEFAULT_WIDTH = 640


DEFAULT_HEIGHT = 512


RESOLUTION_OPTIONS = [
    (256, 192),
    (320, 240),
    (384, 288),
    (640, 512),
    (1024, 768),
]


RAW_DTYPE = "<u2"       # RAW16 Little Endian


DEFAULT_BIT_DEPTH = 14


BIT_DEPTH_OPTIONS = [8, 10, 12, 14, 16]


NOVATEK_NUC_GAIN_SHIFT = 8


NOVATEK_NUC_GAIN_ONE = 1 << NOVATEK_NUC_GAIN_SHIFT


DEFAULT_CAMERA_INDEX = 1


CAMERA_INDEX_OPTIONS = list(range(10))


CAMERA_DEVICE_ID_ROLE = Qt.UserRole + 1


CAMERA_FRIENDLY_NAME_ROLE = Qt.UserRole + 2


CAMERA_REFRESH_INTERVAL_MS = 1000


DEFAULT_UVC_FPS = 30


UVC_CAPTURE_FOURCC = "MJPG"


DEFAULT_UVC_CAPTURE_MODE = "auto_detect"


UVC_CAPTURE_MODE_OPTIONS = [
    ("Auto Detect (MJPG -> Generic)", "auto_detect"),
    ("Novatek Thermal (MJPG Strict)", "novatek_mjpg"),
    ("Generic Webcam (Auto Format)", "generic_auto"),
]


UVC_AUTO_DETECT_FRAME_TIMEOUT_SECONDS = 2.0


UVC_FRAME_WAIT_SECONDS = 0.5


UVC_RECONNECT_SECONDS = 5.0


UVC_CLOSE_TIMEOUT_MS = 3000


UVC_RECORDING_FOURCC = "MJPG"


UVC_RECORDING_QUEUE_SIZE = 8


UVC_VIEWER_ANALYSIS_INTERVAL_SECONDS = 0.5


DUAL_UVC_QUALITY_INTERVAL_SECONDS = 0.5


UVC_QUALITY_ANALYSIS_MAX_PIXELS = 192 * 144


UVC_TEMPORAL_ANALYSIS_MAX_PIXELS = 160 * 120


UVC_TEMPORAL_SAMPLE_INTERVAL_SECONDS = 0.10


UVC_TEMPORAL_WINDOW_SECONDS = 2.5


UVC_TEMPORAL_MIN_DURATION_SECONDS = 1.5


UVC_TEMPORAL_MIN_SAMPLES = 8


DEFAULT_SPATIAL_QUALITY_WEIGHTS = {
    "detail_clarity_score": 22,
    "contrast_quality_score": 18,
    "flat_cleanliness_score": 18,
    "thermal_uniformity_score": 16,
    "clipping_control_score": 10,
    "artifact_control_score": 16,
}


DEFAULT_UVC_TEMPORAL_WEIGHTS = {
    "spatial_quality_score": 70,
    "flicker_control_score": 15,
    "temporal_stability_score": 15,
}


QUALITY_TIMELINE_METRICS = (
    ("Total Score", "overall_score"),
    ("Spatial Quality", "spatial_quality_score"),
    ("Detail Clarity", "detail_clarity_score"),
    ("Contrast Quality", "contrast_quality_score"),
    ("Flat Cleanliness", "flat_cleanliness_score"),
    ("Thermal Uniformity", "thermal_uniformity_score"),
    ("Clipping Control", "clipping_control_score"),
    ("Artifact Control", "artifact_control_score"),
    ("Flicker Control", "flicker_control_score"),
    ("Temporal Stability", "temporal_stability_score"),
)


ACTIVE_SPATIAL_QUALITY_WEIGHTS = dict(DEFAULT_SPATIAL_QUALITY_WEIGHTS)


ACTIVE_UVC_TEMPORAL_WEIGHTS = dict(DEFAULT_UVC_TEMPORAL_WEIGHTS)


DUAL_UVC_IMAGE_NAME_OPTIONS = [
    ("PEGA", "PEGA"),
    ("TianYuan", "TianYuan"),
    ("Viewsec", "Viewsec")
]
