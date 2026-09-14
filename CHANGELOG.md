# Thermal RAW Toolkit

## v1.4.1 - 2026-09-14

- Replace Image/UVC DDE with the firmware Y8 Domain Transform algorithm; expose Sigma S, Sigma R, Gain (%) and Iterations (defaults 8/16/150/3), without NLM or percentile stretching.

- Add configurable Median and Gaussian filters to both Image Algorithm Compare and UVC Algorithm Compare pipelines.
- Replace Image Algorithm Compare's implicit Input Mapping rules with one explicit native-DN to 8-bit Tone Mapping stage.
- Add live, method-specific controls for Linear Full Range, Percentile, native-DN Global Histogram Equalization, native-DN CLAHE and Fieldscale.
- Make native-DN CLAHE bit-depth aware: RAW14 is left-aligned in `uint16`, OpenCV uses 256 effective histogram bins through `bitShift=8`, and Clip Limit keeps its familiar 8-bit-equivalent scale.
- Increase the Image Algorithm Compare `8-bit Enhancement Pipeline` heading size and synchronize Image Quality metric A/B headers with Automatic, preset and Custom image names, including Right Preview stage changes.
- Keep every following Image Algorithm Pipeline step in the 8-bit domain, show the calculated mapping range immediately and disable Save when mapping parameters are invalid.
- Export `reference_linear.png`, method-named `tone_mapped_*.png` files and domain/parameter/calculated-value metadata in `pipeline.json` format version 3.
- Refresh Camera lists from the currently connected devices only; a removed USB camera no longer leaves a stale index, name or disconnected placeholder.

## v1.4.0 ? 2026-09-10

- Dual Image Compare ? Image Algorithm Compare ?? Side by Side?Swipe Compare?????????????
- ????????????? A?B ?????????????????????????????
- ?????????????????????????????????????????????
- ?? Custom... ?????????????????????????? Automatic ???? Right Preview ???????
- ????????????? Save Current Preview PNG ?????????????????????????
- ??????? image_comparison.png??? Right Preview ?????JSON ????????????????????
- ???????????? EXE ????????????????????
- ????? PyInstaller ??????? v1.4.0?

???Python ???Qt ??????????????????????????????????? EXE?????????????

## v1.3.0 — 2026-09-09

- Add persistent horizontal mirror controls for Camera A and Camera B. Camera A mirroring also applies to Single UVC Statistic and UVC Algorithm Compare.
- Export original RAW DN matrices as CSV from the three Image save workflows.
- Add optional filled white BPR point highlights with independent bright-point and red-circle radius controls, live preview, marked-image output and summary metadata.

Validation: Python syntax, settings round-trip, GUI construction, mirror routing and BPR marker rendering checks passed. No EXE was built; physical-camera behavior remains unverified.

## v1.2.0 — 2026-09-07

Changes relative to v1.1.0:

- Rename the RAW category to Image, with Single Image Statistics, Dual Image Compare and Image Algorithm Compare. Add JPG/JPEG, PNG, BMP and TIFF decoding to grayscale with automatic dimensions/depth, preserving 16-bit grayscale input and existing RAW/BIN validation.
- Add full image quality analysis, lightweight UVC quality analysis, temporal quality tracking, histograms/CDF, comparison metrics, quality help and Excel/JSON/text reports. Highlight the higher displayed score in blue and bold; ties and unavailable scores remain neutral.
- Add Toolkit settings for RAW input, output location, log font, timeline and quality weights, with local JSON persistence. Bundle the quality-score guide and add openpyxl to requirements.
- Expand Fieldscale parameters and DDE processing. General-image Fieldscale inputs use edge padding when grid dimensions do not divide evenly, then crop back to the original size.
- Start pipelines empty. Separate Reset All Parameters, Reset Step and Reset Pipeline. Empty Image pipelines support immediate preview and saving.
- Make Image Input Mapping source-aware: ordinary images bypass optional mapping; a first enabled Fieldscale or Percentile Auto Contrast step receives native values. Restore the user's RAW mapping preference when applicable and avoid duplicate percentile mapping.
- Add Single/Dual UVC statistics and comparison workflows, optional quality calculation, live quality timelines, recording/capture reports and image-name selection.
- Add resizable image comparison windows with aspect-ratio preservation, live updates and named comparison PNG export. Synchronize Dual Image names across previews, metrics and reports. Ignore UVC preview double-clicks when no frames are available.
- Introduce Category/Function menu navigation, maximize the window at startup, refine category typography and current-function highlighting, and use action-specific button labels. Disable Image save actions until inputs load successfully.
- Use Image Quality labels and source-specific Image/RAW preview titles and Information messages. Always save the Average RAW Folder preview PNG and restyle the BPR background checkbox.
- Update the PyInstaller specification to v1.2.0 and bundle the quality guide. Replace the removed legacy README with these release notes.

Validation: Python syntax/import checks and targeted GUI/array/export checks were performed during development. The release check covers navigation, initial save states, source decoding, empty pipelines, native percentile mapping, Fieldscale padding and report export. No EXE was built; physical-camera streaming/recording and Novatek board validation are not release-tested here.
