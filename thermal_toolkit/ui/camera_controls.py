"""ui / camera_controls."""

from thermal_toolkit.core.constants import CAMERA_DEVICE_ID_ROLE, CAMERA_FRIENDLY_NAME_ROLE, DEFAULT_CAMERA_INDEX
from thermal_toolkit.uvc.devices import enumerate_camera_devices


class CameraControlsMixin:
    """ui / camera_controls behavior composed by the main window."""

    @staticmethod
    def camera_combo_device_id(combo):
        return combo.currentData(CAMERA_DEVICE_ID_ROLE)

    @staticmethod
    def camera_combo_friendly_name(combo):
        name = combo.currentData(CAMERA_FRIENDLY_NAME_ROLE)
        return str(name) if name else combo.currentText()

    @staticmethod
    def populate_camera_combo(
        combo,
        devices,
        default_index,
        preferred_device_id=None,
        preferred_name=None,
    ):
        combo.blockSignals(True)
        try:
            combo.clear()
            selected_item = -1
            for device in devices:
                camera_index = int(device["index"])
                friendly_name = str(device["name"])
                device_id = str(device["device_id"])
                combo.addItem(
                    f"{friendly_name} (Camera {camera_index})",
                    camera_index,
                )
                item_index = combo.count() - 1
                combo.setItemData(
                    item_index, device_id, CAMERA_DEVICE_ID_ROLE
                )
                combo.setItemData(
                    item_index, friendly_name, CAMERA_FRIENDLY_NAME_ROLE
                )
                if preferred_device_id and device_id == preferred_device_id:
                    selected_item = item_index

            if combo.count() == 0:
                combo.addItem("No camera detected", None)
                combo.setItemData(
                    0, "", CAMERA_DEVICE_ID_ROLE
                )
                combo.setItemData(
                    0, "No camera detected", CAMERA_FRIENDLY_NAME_ROLE
                )
                selected_item = 0
            elif selected_item < 0:
                selected_item = combo.findData(default_index)
                if selected_item < 0:
                    selected_item = 0
            combo.setCurrentIndex(selected_item)
        finally:
            combo.blockSignals(False)

    def selected_camera_info(self, combo):
        camera_index = combo.currentData()
        return {
            "index": None if camera_index is None else int(camera_index),
            "device_id": self.camera_combo_device_id(combo),
            "name": self.camera_combo_friendly_name(combo),
        }

    def refresh_camera_devices(self, _checked=False, log_changes=True):
        devices = enumerate_camera_devices()
        signature = tuple(
            (
                int(device["index"]),
                str(device["device_id"]),
                str(device["name"]),
            )
            for device in devices
        )
        if signature == self._camera_device_signature:
            return

        old_devices = {
            str(device["device_id"]): str(device["name"])
            for device in self.camera_devices
        }
        new_devices = {
            str(device["device_id"]): str(device["name"])
            for device in devices
        }
        selections = []
        for combo in (
            self.viewer_uvc_camera_combo,
            self.uvc_camera_combo,
            self.dual_uvc_camera_a,
            self.dual_uvc_camera_b,
        ):
            selections.append(
                (
                    self.camera_combo_device_id(combo),
                    self.camera_combo_friendly_name(combo),
                )
            )

        self.populate_camera_combo(
            self.viewer_uvc_camera_combo,
            devices,
            DEFAULT_CAMERA_INDEX,
            *selections[0],
        )
        self.populate_camera_combo(
            self.uvc_camera_combo,
            devices,
            DEFAULT_CAMERA_INDEX,
            *selections[1],
        )
        self.populate_camera_combo(
            self.dual_uvc_camera_a,
            devices,
            DEFAULT_CAMERA_INDEX,
            *selections[2],
        )
        default_b_index = 0 if DEFAULT_CAMERA_INDEX != 0 else 1
        self.populate_camera_combo(
            self.dual_uvc_camera_b,
            devices,
            default_b_index,
            *selections[3],
        )

        self.camera_devices = devices
        self._camera_device_signature = signature
        devices_by_id = {
            str(device["device_id"]): device for device in devices
        }
        for worker in (
            self.viewer_uvc_worker,
            self.uvc_worker,
            self.dual_uvc_worker_a,
            self.dual_uvc_worker_b,
        ):
            if worker is None or not worker.camera_device_id:
                continue
            device = devices_by_id.get(str(worker.camera_device_id))
            if device is None:
                worker.update_camera_device(None)
            else:
                worker.update_camera_device(
                    int(device["index"]), str(device["name"])
                )

        if log_changes and hasattr(self, "log"):
            added = [
                new_devices[device_id]
                for device_id in new_devices.keys() - old_devices.keys()
            ]
            removed = [
                old_devices[device_id]
                for device_id in old_devices.keys() - new_devices.keys()
            ]
            if added:
                self.log.append("Camera connected: " + ", ".join(added))
            if removed:
                self.log.append("Camera disconnected: " + ", ".join(removed))
