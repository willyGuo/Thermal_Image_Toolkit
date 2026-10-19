"""uvc / devices."""

from thermal_toolkit.core.constants import CAMERA_INDEX_OPTIONS
from thermal_toolkit.core.qt import QCameraInfo


def camera_device_id_text(device_name):
    if isinstance(device_name, str):
        return device_name
    try:
        return bytes(device_name).decode("utf-8", errors="replace")
    except (TypeError, ValueError):
        return str(device_name)


def enumerate_camera_devices():
    """Return camera index/name/stable ID in Qt multimedia enumeration order."""
    if QCameraInfo is not None:
        try:
            devices = []
            for camera_index, camera_info in enumerate(
                QCameraInfo.availableCameras()
            ):
                friendly_name = str(camera_info.description()).strip()
                device_id = camera_device_id_text(camera_info.deviceName())
                devices.append(
                    {
                        "index": camera_index,
                        "name": friendly_name or f"Camera {camera_index}",
                        "device_id": device_id or f"camera-index:{camera_index}",
                    }
                )
            return devices
        except Exception:
            pass

    return [
        {
            "index": camera_index,
            "name": f"Camera {camera_index}",
            "device_id": f"camera-index:{camera_index}",
        }
        for camera_index in CAMERA_INDEX_OPTIONS
    ]
