"""Three-method bad pixel calibration and SIE export."""
import json
import shutil
from pathlib import Path
import numpy as np
from thermal_toolkit.algorithms.mapping import raw_to_display_u8
from thermal_toolkit.core.qt import QLabel, Qt
from thermal_toolkit.features.bpr.calibration import calibrate_bad_pixels, temporal_statistics, encode_dpc_table, DPC_MAX_PIXELS, OVERLAP_ROWS, overlap_statistics
from thermal_toolkit.features.bpr.render import draw_bpr_markers
from thermal_toolkit.features.bpr.calibration_export import save_bad_pixel_workbook
from thermal_toolkit.io.raw import load_raw
from thermal_toolkit.io.export import save_png, save_rgb_png
from thermal_toolkit.ui.widgets.inputs import PathRow, FocusWheelDoubleSpinBox, FocusWheelSpinBox
from thermal_toolkit.ui.widgets.groups import QualityScoreCheckBox


class BadPixelCalibrationPageMixin:
    def build_bad_pixel_calibration_page(self):
        page, form = self._form_page()
        filt = "RAW Files (*.raw *.RAW *.bin *.BIN)"
        self.bad_pixel_raw_a = PathRow("file", filt, "Uniform blackbody RAW at temperature A")
        self.bad_pixel_raw_b = PathRow("file", filt, "Uniform blackbody RAW at temperature B")
        self.bad_pixel_temporal = PathRow("folder", dialog_title="Same-temperature RAW sequence (100 recommended)")
        for title, row in (("Temperature A RAW (DC / response)", self.bad_pixel_raw_a),
                           ("Temperature B RAW (response)", self.bad_pixel_raw_b),
                           ("NETD temporal RAW folder", self.bad_pixel_temporal)):
            form.addRow(title, row)
            row.edit.textChanged.connect(self.invalidate_bad_pixel_calibration)
        def number(name, title, value, low, high, integer=False):
            widget = FocusWheelSpinBox() if integer else FocusWheelDoubleSpinBox()
            widget.setRange(low, high)
            widget.setValue(value)
            setattr(self, name, widget)
            form.addRow(title, widget)
            widget.valueChanged.connect(self.invalidate_bad_pixel_calibration)
            return widget
        number('bad_pixel_temperature_a', 'Temperature A (C)', 25, -100, 1000)
        number('bad_pixel_temperature_b', 'Temperature B (C)', 60, -100, 1000)
        for name, title in (('dc', 'DC Level'), ('response', 'Responsivity'), ('netd', 'NETD')):
            check = QualityScoreCheckBox('Enable ' + title)
            check.setChecked(True)
            setattr(self, 'bad_pixel_' + name, check)
            form.addRow(check)
            check.toggled.connect(self.invalidate_bad_pixel_calibration)
        self.bad_pixel_statistics, stats_layout = self.create_quality_score_container(
            'Bad Pixel Statistics')
        stats_layout.setHorizontalSpacing(10)
        stats_layout.setVerticalSpacing(7)
        for column, title in enumerate(('Method', 'Status', 'Pixels', 'Ratio')):
            header = QLabel(title)
            header.setStyleSheet('font-weight:700; color:#c9e2f2;')
            header.setAlignment(Qt.AlignLeft if column == 0 else Qt.AlignRight)
            stats_layout.addWidget(header, 0, column)
        self.bad_pixel_statistics_labels = {}
        for row, title in enumerate(('DC Level', 'Responsivity', 'NETD', 'Union (unique)'), 1):
            stats_layout.addWidget(QLabel(title), row, 0)
            values = []
            for column in range(1, 4):
                value = QLabel('--')
                value.setProperty('viewerStatValue', True)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setStyleSheet('color:#c9d1d9; font-weight:400;')
                stats_layout.addWidget(value, row, column)
                values.append(value)
            self.bad_pixel_statistics_labels[title] = values
        note = QLabel('Ratio: all image pixels. Union counts overlapping points once.')
        note.setWordWrap(True)
        stats_layout.addWidget(note, 5, 0, 1, 4)
        form.addRow(self.bad_pixel_statistics)
        overlap_group, overlap_layout = self.create_quality_score_container('Method Overlap')
        overlap_layout.setHorizontalSpacing(10)
        overlap_layout.setVerticalSpacing(7)
        for col, text in enumerate(('Combination', 'Pixels', 'Ratio')):
            header = QLabel(text)
            header.setStyleSheet('font-weight:700; color:#c9e2f2;')
            overlap_layout.addWidget(header, 0, col)
        self.bad_pixel_overlap_labels = {}
        for row, (title, _) in enumerate(OVERLAP_ROWS, 1):
            caption = QLabel(title)
            caption.setWordWrap(True)
            overlap_layout.addWidget(caption, row, 0)
            labels = []
            for col in (1, 2):
                value = QLabel('--')
                value.setProperty('viewerStatValue', True)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                value.setStyleSheet('color:#c9d1d9; font-weight:400;')
                overlap_layout.addWidget(value, row, col)
                labels.append(value)
            self.bad_pixel_overlap_labels[title] = labels
        note = QLabel('First 7 rows sum to Union. Last 3 pairs include All three; do not add them. Ratio: all pixels. N/A: method not enabled.')
        note.setWordWrap(True)
        overlap_layout.addWidget(note, 11, 0, 1, 3)
        form.addRow(overlap_group)

        self.bad_pixel_calibration_status = QLabel('Select inputs and methods, then press Run.')
        self.bad_pixel_calibration_status.setWordWrap(True)
        form.addRow('Result', self.bad_pixel_calibration_status)
        self.resolution_combo.currentIndexChanged.connect(self.invalidate_bad_pixel_calibration)
        self.bit_depth_combo.currentIndexChanged.connect(self.invalidate_bad_pixel_calibration)
        self.stack.addWidget(page)
        self.update_bad_pixel_inputs()
        self.update_bad_pixel_statistics()

    def update_bad_pixel_statistics(self, counts=None, total=None, union=None, overlaps=None):
        rows = []
        for name, title in (("dc", "DC Level"), ("response", "Responsivity"), ("netd", "NETD")):
            enabled = getattr(self, "bad_pixel_" + name).isChecked()
            state = "Done" if counts is not None and enabled else ("Pending" if enabled else "Off")
            count = f"{counts[title]:,}" if counts is not None and enabled else "-"
            ratio = f"{100 * counts[title] / total:.3f}%" if counts is not None and enabled else "-"
            rows.append((title, state, count, ratio))
        rows.append(("Union (unique)", "Done" if counts is not None else "Pending",
                     f"{union:,}" if union is not None else "-",
                     f"{100 * union / total:.3f}%" if union is not None else "-"))
        for title, state, count, ratio in rows:
            for label, text in zip(self.bad_pixel_statistics_labels[title], (state, count, ratio)):
                label.setText('--' if text == '-' else text)

        for title, keys in OVERLAP_ROWS:
            enabled = all(getattr(self, 'bad_pixel_' + key).isChecked() for key in keys)
            values = ('--', '--') if overlaps is None else (
                (f'{overlaps[title]:,}', f'{100 * overlaps[title] / total:.3f}%') if enabled else ('N/A', 'N/A'))
            for label, text in zip(self.bad_pixel_overlap_labels[title], values):
                label.setText(text)

    def update_bad_pixel_inputs(self):
        needs_b = self.bad_pixel_response.isChecked() or self.bad_pixel_netd.isChecked()
        self.bad_pixel_raw_b.setEnabled(needs_b)
        self.bad_pixel_temporal.setEnabled(self.bad_pixel_netd.isChecked())
        for name in ('temperature_a', 'temperature_b'):
            getattr(self, 'bad_pixel_' + name).setEnabled(self.bad_pixel_netd.isChecked())

    def invalidate_bad_pixel_calibration(self, _value=None):
        if not hasattr(self, 'bad_pixel_calibration_status'):
            return
        self.update_bad_pixel_inputs()
        self.update_bad_pixel_statistics()
        self.bad_pixel_calibration_status.setToolTip('')
        self.bad_pixel_calibration_status.setText('Select inputs and methods, then press Run.')
        if self.FUNCTIONS[self.current_function_index()] == 'Bad pixel calibration':
            self.current_original = self.current_result = None
            self.preview_original.set_array(None)
            self.preview_result.set_array(None)

    def run_bad_pixel_calibration(self):
        self.invalidate_bad_pixel_calibration()
        path_a, path_b = self.bad_pixel_raw_a.text(), self.bad_pixel_raw_b.text()
        needs_b = self.bad_pixel_response.isChecked() or self.bad_pixel_netd.isChecked()
        if not path_a:
            raise ValueError('Select temperature A RAW.')
        if needs_b and not path_b:
            raise ValueError('Responsivity / NETD requires temperature B RAW.')
        if not needs_b:
            path_b = None
        loader = lambda path: load_raw(str(path), self.width, self.height)
        a, b = loader(path_a), loader(path_b) if needs_b else None
        params = {name: getattr(self, 'bad_pixel_' + name).value() for name in
                  ('temperature_a', 'temperature_b')}
        params.update(self.app_settings['bad_pixel_calibration'])
        params.update({name: getattr(self, 'bad_pixel_' + name).isChecked() for name in ('dc', 'response', 'netd')})
        paths, temporal_std = [], None
        if params['netd']:
            folder = Path(self.bad_pixel_temporal.text())
            if not self.bad_pixel_temporal.text() or not folder.is_dir():
                raise ValueError('Select the NETD same-temperature RAW folder.')
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.raw', '.bin'))
            _, temporal_std = temporal_statistics(paths, loader)
        result = calibrate_bad_pixels(a, b, temporal_std=temporal_std, **params)
        coordinates = result['coordinates']
        table = None
        skip_reason = None
        if len(coordinates) > DPC_MAX_PIXELS:
            skip_reason = (f'{len(coordinates):,} candidates exceed the '
                           f'{DPC_MAX_PIXELS:,}-pixel SIE limit; no truncation.')
        else:
            table = encode_dpc_table(coordinates)
        table_status = ('dpc_table.bin saved' if table is not None else
                        'dpc_table.bin NOT generated: ' + skip_reason)
        out = self.make_output('bad_pixel_calibration')
        a.astype('<u2', copy=False).tofile(out / 'temperature_a.raw')
        if b is not None:
            b.astype('<u2', copy=False).tofile(out / 'temperature_b.raw')
        if paths:
            target = out / 'temporal_inputs'
            target.mkdir()
            for i, path in enumerate(paths):
                shutil.copyfile(path, target / f'{i:04d}_{path.name}')
        if table is not None:
            (out / 'dpc_table.bin').write_bytes(table)
        counts = {name: int(mask.sum()) for name, mask in result['masks'].items()}
        overlap_counts = {}
        for x, y in coordinates:
            label = '; '.join(k for k, mask in result['masks'].items() if mask[y, x])
            overlap_counts[label] = overlap_counts.get(label, 0) + 1
        metadata = dict(total_pixels=int(a.size),
                        union_percent=100 * len(coordinates) / a.size,
                        method_percent={k: 100 * v / a.size for k, v in counts.items()},
                        method_combination_counts=overlap_counts,
                        overlap_statistics=overlap_statistics(result['masks']),
                        dpc_table_generated=table is not None,
                        dpc_table_skip_reason=skip_reason, dpc_max_pixels=DPC_MAX_PIXELS,
                        csv_threshold_convention='Compared magnitude > threshold; excess=max(0, magnitude-threshold). Response excess uses percentage points. Disabled methods are N/A. NETD invalid slopes have no numeric excess.',
                        parameters=params, source_a=path_a, source_b=path_b,
                        temporal_sources=[str(p) for p in paths], temporal_count=len(paths),
                        method_counts=counts, union_count=len(coordinates),
                        response_reference_dn=result['response_reference_dn'],
                        response_reference_note='Global response median is used for NETD validity only; Responsivity uses each pixel local 9x9 median, centre excluded, valid border neighbours.',
                        response_window=9,
                        netd_mean_mk=result['netd_mean_mk'], netd_std_mk=result['netd_std_mk'],
                        coordinate_convention='Top-left (0,0); x right, y down; no mirror',
                        dc_neighbours='Valid neighbours, centre excluded; population std',
                        netd_reference='Finite positive-relative-response slopes only; invalid slopes flagged')
        (out / 'calibration_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        save_bad_pixel_workbook(out / 'bad_pixel_coordinates.xlsx', result, params, a, b, metadata)
        summary = ('Methods: ' + str(counts) + '\nUnion: ' + str(len(coordinates)) +
                   f' / {a.size} ({100 * len(coordinates) / a.size:.4f}%)' +
                   '\nMethod combinations: ' + str(overlap_counts) +
                   '\n' + table_status + '\nNETD frames: ' + str(len(paths)) + '\nSee calibration_summary.json for parameters and sources.\n')
        (out / 'calibration_summary.txt').write_text(summary, encoding='utf-8')
        self.current_original = raw_to_display_u8(a, self.raw_max)
        self.current_result = draw_bpr_markers(raw_to_display_u8(b if b is not None else a, self.raw_max), coordinates)
        save_png(out / 'preview_original.png', self.current_original)
        save_rgb_png(out / 'preview_bad_pixels.png', self.current_result)
        comparison = np.concatenate(
            (np.repeat(self.current_original[:, :, None], 3, axis=2), self.current_result),
            axis=1,
        )
        save_rgb_png(out / 'preview_comparison.png', comparison)
        self.preview_original.set_array(self.current_original)
        self.preview_result.set_array(self.current_result)
        self.update_bad_pixel_statistics(counts, a.size, len(coordinates), metadata['overlap_statistics'])
        self.bad_pixel_calibration_status.setText(
            f'Total: {len(coordinates):,} / {a.size:,}\n{table_status}')
        self.bad_pixel_calibration_status.setToolTip(f'Saved results: {out}')
        self.log.append(summary)
        self.log.append(f'Saved results: {out}')
