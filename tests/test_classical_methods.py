"""Unit tests for src/classical_methods.py (CHROM, POS).

These tests build a synthetic RGB video signal with a known, injected
heart-rate-frequency modulation and check that both classical methods
recover the correct BPM.
"""

import numpy as np
import pytest

from src.classical_methods import chrom, pos
from src.utils import estimate_hr_fft


def make_synthetic_rgb(bpm: float, fps: float, duration_sec: float, seed: int = 0) -> np.ndarray:
    """Simulate per-frame mean R, G, B values from a facial ROI with a
    weak pulsatile component riding on a much larger DC skin-tone level,
    matching the physical model CHROM/POS are derived from: the green
    channel carries the strongest plethysmographic signal, red a weaker
    one, and blue weaker still, all sharing the same underlying pulse.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fps)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)

    # Baseline skin-tone levels (arbitrary but realistic-ish 8-bit-ish scale).
    base_r, base_g, base_b = 150.0, 100.0, 80.0
    # Relative pulsatile amplitudes per channel (green strongest -- standard
    # assumption underlying rPPG).
    r = base_r + 0.6 * pulse + rng.normal(0, 0.05, n)
    g = base_g + 1.0 * pulse + rng.normal(0, 0.05, n)
    b = base_b + 0.3 * pulse + rng.normal(0, 0.05, n)

    # Add a slow illumination drift shared across channels (simulates
    # ambient lighting change / minor head motion) -- CHROM/POS should
    # be robust to this since it cancels in the chrominance combination.
    drift = 5.0 * np.sin(2 * np.pi * 0.05 * t)
    r, g, b = r + drift, g + drift, b + drift

    return np.stack([r, g, b], axis=1)


class TestChrom:
    @pytest.mark.parametrize("bpm", [55, 70, 90, 110])
    def test_recovers_known_bpm(self, bpm):
        fps = 30.0
        rgb = make_synthetic_rgb(bpm=bpm, fps=fps, duration_sec=20, seed=hash(bpm) % 1000)
        pulse_signal = chrom(rgb, fps=fps)
        est = estimate_hr_fft(pulse_signal, fps=fps)
        assert abs(est.bpm - bpm) < 3.0, f"CHROM: expected ~{bpm}, got {est.bpm}"

    def test_output_length_matches_input(self):
        fps = 30.0
        rgb = make_synthetic_rgb(bpm=75, fps=fps, duration_sec=10)
        pulse_signal = chrom(rgb, fps=fps)
        assert len(pulse_signal) == len(rgb)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError):
            chrom(np.random.randn(100, 2), fps=30.0)


class TestPos:
    @pytest.mark.parametrize("bpm", [55, 70, 90, 110])
    def test_recovers_known_bpm(self, bpm):
        fps = 30.0
        rgb = make_synthetic_rgb(bpm=bpm, fps=fps, duration_sec=20, seed=hash(bpm) % 1000)
        pulse_signal = pos(rgb, fps=fps)
        est = estimate_hr_fft(pulse_signal, fps=fps)
        assert abs(est.bpm - bpm) < 3.0, f"POS: expected ~{bpm}, got {est.bpm}"

    def test_output_length_matches_input(self):
        fps = 30.0
        rgb = make_synthetic_rgb(bpm=75, fps=fps, duration_sec=10)
        pulse_signal = pos(rgb, fps=fps)
        assert len(pulse_signal) == len(rgb)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError):
            pos(np.random.randn(100, 2), fps=30.0)
