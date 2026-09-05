"""
Classical (non-learned) rPPG methods: CHROM and POS.

These operate on per-frame mean R, G, B values extracted from a facial ROI
and produce a single 1D pulse signal. They serve as the baseline that the
deep-learning model in model.py is compared against.

References:
    CHROM: de Haan, G., & Jeanne, V. (2013). "Robust pulse rate from
        chrominance-based rPPG." IEEE Trans. Biomedical Engineering.
    POS:   Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
        "Algorithmic principles of remote-PPG." IEEE Trans. Biomedical
        Engineering.
"""

from __future__ import annotations

import numpy as np

from src.utils import bandpass_filter, detrend


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Temporally normalize each channel: divide by its own temporal mean.
    This is the standard first step in both CHROM and POS -- it removes
    the (arbitrary) absolute skin-tone/illumination level and leaves only
    relative fluctuations, which is what carries the pulse information.

    Args:
        rgb: (N, 3) array of per-frame mean R, G, B values.

    Returns:
        (N, 3) normalized array.
    """
    means = rgb.mean(axis=0, keepdims=True)
    means[means == 0] = 1e-8
    return rgb / means


def chrom(rgb: np.ndarray, fps: float) -> np.ndarray:
    """CHROM method: builds two chrominance signals that cancel specular
    reflection / motion-induced intensity changes, combines them with a
    ratio of their standard deviations to further suppress residual
    motion, then bandpass-filters the result.

    Args:
        rgb: (N, 3) array of per-frame mean R, G, B values from the ROI.
        fps: Video frame rate in Hz.

    Returns:
        1D pulse signal of length N (bandpassed, zero-mean).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb of shape (N, 3), got {rgb.shape}")

    rn = _normalize_rgb(rgb)
    r, g, b = rn[:, 0], rn[:, 1], rn[:, 2]

    # Standard CHROM chrominance combinations (de Haan & Jeanne, 2013).
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b

    std_x = np.std(x)
    std_y = np.std(y)
    alpha = std_x / (std_y + 1e-8)
    pulse = x - alpha * y

    pulse = detrend(pulse)
    pulse = bandpass_filter(pulse, fps=fps)
    return pulse


def pos(rgb: np.ndarray, fps: float, window_size_sec: float = 1.6) -> np.ndarray:
    """POS (Plane-Orthogonal-to-Skin) method: projects the RGB signal onto
    a plane orthogonal to the skin-tone vector under a sliding window,
    which is more robust to specular reflection than CHROM under
    realistic (non-controlled) lighting.

    Args:
        rgb: (N, 3) array of per-frame mean R, G, B values from the ROI.
        fps: Video frame rate in Hz.
        window_size_sec: POS sliding-window length in seconds
            (1.6s is the value used in the original paper).

    Returns:
        1D pulse signal of length N (bandpassed, zero-mean).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb of shape (N, 3), got {rgb.shape}")

    n = rgb.shape[0]
    window_len = max(3, int(round(window_size_sec * fps)))
    pulse = np.zeros(n)

    # Projection matrix onto the plane orthogonal to the skin-tone vector,
    # expressed in a fixed basis (Wang et al., 2017, Eq. 6).
    proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)

    for start in range(n - window_len + 1):
        end = start + window_len
        segment = rgb[start:end]
        c_mean = segment.mean(axis=0)
        c_mean[c_mean == 0] = 1e-8
        c_norm = segment / c_mean  # temporal normalization within window

        s = c_norm @ proj.T  # (window_len, 2) -> [s1, s2]
        s1, s2 = s[:, 0], s[:, 1]
        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + alpha * s2
        h = h - np.mean(h)

        # Overlap-add into the output pulse signal.
        pulse[start:end] += h

    pulse = detrend(pulse)
    pulse = bandpass_filter(pulse, fps=fps)
    return pulse


METHODS = {"CHROM": chrom, "POS": pos}
