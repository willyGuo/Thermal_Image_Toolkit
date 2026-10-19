"""core / paths."""

from datetime import datetime
from pathlib import Path
import sys


def bundled_resource_path(filename):
    source_root = Path(__file__).resolve().parents[2]
    source_path = source_root / filename
    if not getattr(sys, "frozen", False):
        return source_path

    external_path = Path(sys.executable).resolve().parent / filename
    if external_path.is_file():
        return external_path

    return Path(getattr(sys, "_MEIPASS", source_root)) / filename


def toolkit_root_path():
    return (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parents[2]
    )


def toolkit_settings_path():
    return toolkit_root_path() / "docs" / "toolkit_settings.json"


def default_output_root():
    return str(toolkit_root_path() / "Thermal_RAW_Results")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def unique_result_dir(base_dir, function_name):
    root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd() / "Thermal_RAW_Results"
    function_root = root / function_name
    function_root.mkdir(parents=True, exist_ok=True)
    out = function_root / timestamp()
    counter = 1
    original = out
    while out.exists():
        out = Path(f"{original}_{counter:02d}")
        counter += 1
    out.mkdir(parents=True, exist_ok=False)
    return out
