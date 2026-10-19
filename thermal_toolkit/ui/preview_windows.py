"""ui / preview_windows."""

import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8, signed_delta_to_display_u8
from thermal_toolkit.core.opencv import cv2
from thermal_toolkit.core.qt import QImage, QMessageBox, QPixmap
from thermal_toolkit.io.export import save_png, save_rgb_png
from thermal_toolkit.reports.pipeline import save_uvc_comparison_png
from thermal_toolkit.ui.widgets.preview import SwipeCompareWidget, UvcVideoWindow
from thermal_toolkit.uvc.recording import UvcRecordingThread


class PreviewWindowsMixin:
    """ui / preview_windows behavior composed by the main window."""

    def current_uvc_recording_view(self):
        function_index = self.current_function_index()
        viewer_uvc_page_index = self.FUNCTIONS.index("Single UVC Statistic")
        dual_uvc_page_index = len(self.FUNCTIONS) - 3
        uvc_page_index = len(self.FUNCTIONS) - 1
        if function_index == viewer_uvc_page_index:
            if self.viewer_uvc_frame is None:
                return None, None
            return (
                cv2.cvtColor(
                    self.viewer_uvc_frame, cv2.COLOR_GRAY2BGR
                ),
                "Single UVC Statistic",
            )
        if function_index == dual_uvc_page_index:
            if self.dual_uvc_frame_a is None or self.dual_uvc_frame_b is None:
                return None, None
            label_a, label_b = self.dual_uvc_labels()
            return (
                UvcRecordingThread._compose_frame(
                    self.dual_uvc_frame_a,
                    self.dual_uvc_frame_b,
                    label_a,
                    label_b,
                ),
                "Dual UVC Recording View",
            )
        if function_index == uvc_page_index:
            if self.uvc_original_frame is None or self.uvc_processed_frame is None:
                return None, None
            return (
                UvcRecordingThread._compose_frame(
                    self.uvc_original_frame,
                    self.uvc_processed_frame,
                    "Original UVC",
                    "Final Pipeline Output",
                ),
                "UVC Pipeline Recording View",
            )
        return None, None

    def open_preview_window(self, preview):
        function_name = self.FUNCTIONS[self.current_function_index()]
        if self.category_name_for_function(function_name) != "Image":
            self.open_uvc_video_window()
            return
        if preview._pixmap is None:
            return
        key = "dual_image" if function_name == "Dual Image Compare" else preview
        if function_name == "Image Algorithm Compare":
            key = "image_algorithm"
        window = self.image_preview_windows.get(key)
        if window is None:
            window = UvcVideoWindow(self)
            if key in ("dual_image", "image_algorithm"):
                window.swipe_preview = SwipeCompareWidget()
                window.layout().addWidget(window.swipe_preview)
                window.swipe_preview.hide()
                window.swipe_preview.position_changed.connect(
                    lambda position, key=key: self.sync_dual_swipe_position(position, key)
                )
            self.image_preview_windows[key] = window
        window.show()
        self.update_image_preview_window(preview)
        window.raise_()
        window.activateWindow()

    def update_image_preview_titles(self):
        name = self.FUNCTIONS[self.current_function_index()]
        def source_label(info):
            return "RAW" if info.get("source_type") == "raw" else "Image"
        if name == "Single Image Statistics":
            kind = source_label(getattr(self, "raw_viewer_info", {}))
            titles = (None, f"{kind} Preview (Fixed Scale)")
        elif name == "Image Algorithm Compare":
            kind = source_label(getattr(self, "raw_algorithm_info", {}))
            titles = self.algorithm_image_names()
        elif name == "Dual Image Compare":
            names = self.dual_image_names()
            infos = getattr(self, "dual_image_source_infos", ({}, {}))
            titles = tuple(f"{label} [{source_label(info)}]" for label, info in zip(names, infos))
        else:
            return
        for preview, caption, title in zip(
            (self.preview_original, self.preview_result),
            (self.preview_original_caption, self.preview_result_caption), titles
        ):
            if title is not None:
                preview.set_title(title)
                caption.setText(title)

    def image_comparison_key(self):
        return "image_algorithm" if self.FUNCTIONS[self.current_function_index()] == "Image Algorithm Compare" else "dual_image"

    def image_comparison_mode(self):
        return self.algorithm_compare_mode if self.image_comparison_key() == "image_algorithm" else self.dual_compare_mode

    def sync_dual_swipe_position(self, position, key=None):
        key = key or self.image_comparison_key()
        if not hasattr(self, "image_swipe_positions"):
            self.image_swipe_positions = {}
        self.image_swipe_positions[key] = position
        if key == self.image_comparison_key():
            self.dual_swipe_preview.set_position(position)
        window = self.image_preview_windows.get(key)
        if window is not None:
            window.swipe_preview.set_position(position)

    def update_dual_compare_mode(self, _index=None):
        name = self.FUNCTIONS[self.current_function_index()]
        is_comparison = name in ("Dual Image Compare", "Image Algorithm Compare")
        swipe = is_comparison and self.image_comparison_mode().currentIndex() == 1
        self.dual_compare_controls.setVisible(name == "Dual Image Compare")
        self.dual_swipe_preview.setVisible(swipe)
        self.dual_swipe_preview.set_position(
            getattr(self, "image_swipe_positions", {}).get(self.image_comparison_key(), 0.5)
        )
        show_original = self.PREVIEW_LAYOUTS[self.current_function_index()][0]
        self.preview_original_container.setVisible(show_original and not swipe)
        self.preview_result_container.setVisible(not swipe)
        if is_comparison:
            self.update_image_preview_window(self.preview_original)

    def save_dual_swipe_preview(self, path):
        # Render at source canvas resolution, independent of preview window size.
        export = SwipeCompareWidget()
        export.setMinimumSize(1, 1)
        export.resize(max(frame.width() for frame in self.dual_swipe_preview.frames),
                      max(frame.height() for frame in self.dual_swipe_preview.frames))
        export.set_frames(self.dual_swipe_preview.frames, self.dual_swipe_preview.names)
        export.set_position(self.dual_swipe_preview.position)
        pixmap = QPixmap(export.size())
        export.render(pixmap)
        if not pixmap.save(str(path), "PNG"):
            raise OSError(f"Could not save comparison: {path}")

    def compose_dual_image_preview(self, frames, names=None):
        panel_height = max(frame.shape[0] for frame in frames)
        panel_width = max(frame.shape[1] for frame in frames)
        fitted_frames = []
        for frame in frames:
            scale = min(panel_width / frame.shape[1], panel_height / frame.shape[0])
            width = max(1, min(panel_width, round(frame.shape[1] * scale)))
            height = max(1, min(panel_height, round(frame.shape[0] * scale)))
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((panel_height, panel_width), dtype=np.uint8)
            x = (panel_width - width) // 2
            y = (panel_height - height) // 2
            canvas[y:y + height, x:x + width] = resized
            fitted_frames.append(canvas)
        return UvcRecordingThread._compose_frame(
            *fitted_frames, *(self.dual_image_names() if names is None else names)
        )

    def update_image_preview_window(self, preview):
        function_name = self.FUNCTIONS[self.current_function_index()]
        is_dual = function_name == "Dual Image Compare"
        is_algorithm = function_name == "Image Algorithm Compare"
        if is_dual or is_algorithm:
            pixmaps = (self.preview_original._pixmap, self.preview_result._pixmap)
            names = (self.preview_original.title, self.preview_result.title)
            self.dual_swipe_preview.set_frames(pixmaps, names)
        key = "image_algorithm" if is_algorithm else "dual_image" if is_dual else preview
        window = self.image_preview_windows.get(key)
        if window is None or not window.isVisible():
            return
        if is_dual or is_algorithm:
            swipe = self.image_comparison_mode().currentIndex() == 1
            window.image_label.setVisible(not swipe)
            window.swipe_preview.setVisible(swipe)
            window.swipe_preview.set_frames(pixmaps, names)
            window.swipe_preview.set_position(self.dual_swipe_preview.position)
            window.setWindowTitle(function_name)
            if swipe:
                return
        if is_dual or is_algorithm:
            frames = []
            for panel in (self.preview_original, self.preview_result):
                if panel._pixmap is None:
                    window.clear_frame()
                    return
                image = panel._pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
                buffer = image.bits()
                buffer.setsize(image.sizeInBytes())
                frames.append(
                    np.frombuffer(buffer, dtype=np.uint8)
                    .reshape(image.height(), image.bytesPerLine())[:, :image.width()].copy()
                )
            window.setWindowTitle(function_name)
            names = (
                (self.preview_original.title, self.preview_result.title)
                if is_algorithm else self.dual_image_names()
            )
            window.set_bgr_frame(self.compose_dual_image_preview(frames, names))
            return
        if preview._pixmap is None:
            window.clear_frame()
            return
        window.setWindowTitle(f"{function_name} - {preview.title}")
        window._pixmap = QPixmap(preview._pixmap)
        window._rescale()

    def open_uvc_video_window(self):
        function_index = self.current_function_index()
        uvc_page_indexes = {
            self.FUNCTIONS.index("Single UVC Statistic"),
            len(self.FUNCTIONS) - 3,
            len(self.FUNCTIONS) - 1,
        }
        if function_index not in uvc_page_indexes:
            return
        frame, title = self.current_uvc_recording_view()
        if frame is None:
            return
        if self.uvc_video_window is None:
            self.uvc_video_window = UvcVideoWindow(self)
        self.uvc_video_window.setWindowTitle(title)
        self.uvc_video_window.set_bgr_frame(frame)
        self.uvc_video_window.show()
        self.uvc_video_window.raise_()
        self.uvc_video_window.activateWindow()

    def update_uvc_video_window(self):
        window = self.uvc_video_window
        if window is None or not window.isVisible():
            return
        frame, title = self.current_uvc_recording_view()
        if frame is None:
            return
        window.setWindowTitle(title)
        window.set_bgr_frame(frame)

    def close_uvc_video_window(self):
        if self.uvc_video_window is None:
            return
        self.uvc_video_window.hide()
        self.uvc_video_window.clear_frame()

    def open_raw_histogram_window(self):
        if self.current_function_index() != self.FUNCTIONS.index("Single Image Statistics"):
            return
        if self.raw_viewer_histogram_rgb is None:
            QMessageBox.information(
                self,
                "No Histogram",
                "Select a RAW file and wait for the preview first.",
            )
            return
        if self.raw_viewer_histogram_window is None:
            self.raw_viewer_histogram_window = UvcVideoWindow(self)
        self.raw_viewer_histogram_window.setWindowTitle(
            "Single Image Statistics - DN Histogram and CDF"
        )
        self.raw_viewer_histogram_window.set_bgr_frame(
            cv2.cvtColor(
                self.raw_viewer_histogram_rgb, cv2.COLOR_RGB2BGR
            )
        )
        self.raw_viewer_histogram_window.show()
        self.raw_viewer_histogram_window.raise_()
        self.raw_viewer_histogram_window.activateWindow()

    def close_raw_histogram_window(self):
        if self.raw_viewer_histogram_window is None:
            return
        self.raw_viewer_histogram_window.hide()
        self.raw_viewer_histogram_window.clear_frame()

    def save_current_preview(self):
        if self.current_result is None:
            QMessageBox.information(
                self, "No Preview", "Run a function first."
            )
            return
        function_index = self.current_function_index()
        is_uvc_algorithm = (
            self.FUNCTIONS[function_index] == "UVC Algorithm Compare"
        )
        if is_uvc_algorithm:
            if self.uvc_original_frame is None:
                QMessageBox.information(
                    self,
                    "No UVC Frame",
                    "Start the camera and wait for both previews first.",
                )
                return
            out = self.make_output("manual_preview")
            original = self.uvc_original_frame.copy()
            processed = self.current_result.copy()
            algorithm_title = self.uvc_result_title()
            original_path = out / "original_uvc.png"
            processed_path = out / "processed_selected_pipeline.png"
            comparison_path = out / "comparison_selected_pipeline.png"
            save_png(original_path, original)
            save_png(processed_path, processed)
            save_uvc_comparison_png(
                comparison_path, original, processed, algorithm_title
            )
            self.log.append(f"Saved UVC preview: {algorithm_title}")
            self.log.append(f"Saved original: {original_path}")
            self.log.append(f"Saved processed: {processed_path}")
            self.log.append(f"Saved comparison: {comparison_path}")
            return

        out = self.make_output("manual_preview")
        path = out / "preview.png"
        if (self.FUNCTIONS[function_index] in ("Dual Image Compare", "Image Algorithm Compare")
                and self.image_comparison_mode().currentIndex() == 1):
            self.save_dual_swipe_preview(path)
            self.log.append(f"Saved swipe comparison preview: {path}")
            return
        is_bpr_marking = (
            self.FUNCTIONS[function_index] == "BPR Bad Pixel Marking"
        )
        if is_bpr_marking:
            preview_to_save = np.clip(
                self.current_result, 0, 255
            ).astype(np.uint8)
        elif self.FUNCTIONS[function_index] == "Single Image Statistics":
            preview_to_save = raw_to_display_u8(self.current_result, self.raw_viewer_info["value_max"])
        elif function_index == 4:
            preview_to_save = signed_delta_to_display_u8(
                self.current_result, self.raw_max
            )
        elif function_index in (0, 1, 2, 3, 5):
            preview_to_save = raw_to_display_u8(
                self.current_result, self.raw_max
            )
        else:
            preview_to_save = np.clip(
                self.current_result, 0, 255
            ).astype(np.uint8)
        if is_bpr_marking:
            save_rgb_png(path, preview_to_save)
        else:
            save_png(path, preview_to_save)
        self.log.append(f"Saved preview: {path}")
