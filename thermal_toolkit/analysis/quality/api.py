"""analysis / quality / api."""

from thermal_toolkit.analysis.quality.uvc import thermal_uvc_quality_analysis


def thermal_image_quality_analysis(image, max_analysis_pixels=None):
    """Backward-compatible alias for the lightweight 8-bit UVC profile."""
    return thermal_uvc_quality_analysis(image, max_analysis_pixels=max_analysis_pixels)


def quality_analysis_manifest(result):
    manifest = {"metrics": dict(result["metrics"])}
    if "unavailable_reasons" in result:
        manifest["unavailable_reasons"] = dict(result["unavailable_reasons"])
    if "flat_regions" in result:
        manifest["flat_regions"] = list(result["flat_regions"])
    return manifest
