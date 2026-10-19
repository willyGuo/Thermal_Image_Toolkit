"""ui / settings."""

from thermal_toolkit.core.qt import QMessageBox
from thermal_toolkit.core.settings import (
    default_toolkit_settings,
    save_toolkit_settings,
    validated_toolkit_settings,
)
from thermal_toolkit.ui.settings_dialog import ToolkitSettingsDialog
from thermal_toolkit.algorithms.weighted_aci import configure_weighted_aci


class SettingsMixin:
    """ui / settings behavior composed by the main window."""

    def update_log_font_size(self, size):
        self.log.setStyleSheet(f"font-size:{size}px;")
        if hasattr(self, "bpr_coordinate_info"):
            self.bpr_coordinate_info.setStyleSheet(
                f'font-family:"Consolas"; font-size:{size}px;'
            )

    def show_toolkit_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = ToolkitSettingsDialog(
                self.app_settings, self
            )
            self.settings_dialog.settings_applied.connect(
                self.apply_toolkit_settings
            )
        else:
            self.settings_dialog.set_settings(self.app_settings)
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def apply_toolkit_settings(
        self, settings, persist=True, announce=True
    ):
        settings = validated_toolkit_settings(settings)
        previous_bad_pixel = self.app_settings.get('bad_pixel_calibration')
        self.app_settings = settings
        if previous_bad_pixel != settings['bad_pixel_calibration'] and hasattr(self, 'bad_pixel_calibration_status'):
            self.invalidate_bad_pixel_calibration()
        configure_weighted_aci(settings["weighted_mapping"])
        self.sync_image_input_mapping()
        for editor in (self.raw_pipeline_editor, self.uvc_pipeline_editor):
            editor.load_selected_step(editor.stage_list.currentRow())
        self.update_log_font_size(settings["log_font_size"])

        raw_input = settings["raw_input"]
        selected_resolution = tuple(raw_input["resolution"])
        resolution_index = next(
            (
                index
                for index in range(self.resolution_combo.count())
                if tuple(self.resolution_combo.itemData(index))
                == selected_resolution
            ),
            -1,
        )
        if resolution_index >= 0:
            self.resolution_combo.setCurrentIndex(resolution_index)
        bit_depth_index = self.bit_depth_combo.findData(
            raw_input["bit_depth"]
        )
        if bit_depth_index >= 0:
            self.bit_depth_combo.setCurrentIndex(bit_depth_index)
        self.output_root.setText(raw_input["output_root"])

        self.refresh_raw_histogram_cdf()
        self.refresh_uvc_histogram_cdf()
        if getattr(self, "raw_algorithm_source", None) is not None:
            self.refresh_raw_pipeline_preview()
        if hasattr(self, "uvc_pipeline_editor"):
            self.mark_uvc_processing_config_changed()
            self.refresh_uvc_processed_preview()

        settings_saved = True
        if persist:
            try:
                save_toolkit_settings(settings)
            except OSError as exc:
                settings_saved = False
                QMessageBox.warning(
                    self,
                    "Setting Save Failed",
                    f"Setting was applied but could not be saved:\n{exc}",
                )
        if announce:
            self.log.append(
                f"Setting applied: log {settings['log_font_size']} px; "
                f"RAW {self.width} × {self.height}, "
                f"{self.bit_depth}-bit"
            )
            if self.settings_dialog is not None:
                self.settings_dialog.show_apply_status(
                    "Setting applied successfully."
                    if settings_saved
                    else "Setting applied, but the JSON file was not saved.",
                    success=settings_saved,
                )
