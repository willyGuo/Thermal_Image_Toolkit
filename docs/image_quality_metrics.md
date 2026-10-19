# Static image quality metrics

Static image pages use Gradient, Entropy, C_global, C_local and TMQI.
There is no aggregate score or user-editable weighting. UVC scoring, timelines, and quality reports have been removed; preview, capture, recording, and basic statistics remain available.
See [Chinese user guide](../html/image_quality_guide.html) for input/reference semantics,
normalization, metric directions and reasons for unavailable values.

The metric set follows [Fieldscale IV-B](https://arxiv.org/html/2405.15395v1#S4.SS2).
Fieldscale's public repository does not contain its complete evaluation implementation.
Gradient uses normalized 3x3 Sobel magnitude (scale=1/8, OpenCV default reflection border),
and entropy is 256-bin Shannon entropy divided by 8. These are explicit toolkit definitions;
they are not claimed to reproduce Fieldscale Table I exactly.

Contrast preprocessing and equations follow
[HENSOLDT ToneMappingIQA](https://github.com/HensoldtOptronicsCV/ToneMappingIQA),
the implementation accompanying Teutsch et al., CVPR Workshops 2020.
`tmqi_reference.py` adapts its TMQI implementation, retains the MIT notice,
and removes dataset loading and command-line dependencies.
Original TMQI: Yeganeh and Wang, *Objective Quality Assessment of Tone Mapped Images*,
IEEE TIP 22(2), 657–667, 2013. [Author's implementation](https://ece.uwaterloo.ca/~z70wang/research/tmqi/).

The contrast implementation retains float32 preprocessing and OpenCV filtering;
TMQI retains five scales, weights and statistical naturalness from the reference.
Degenerate constant references, undersized inputs, or nonfinite TMQI results produce
N/A with a reason, rather than zero or a modified metric. No smaller image is upscaled.
SciPy is required for TMQI. All metrics run at full resolution.

Saved JSON retains null and unavailable reasons; Excel uses blank metric cells for N/A.
Each metric has a separate chart because scales and preferred directions differ.
