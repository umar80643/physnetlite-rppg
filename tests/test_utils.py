"""Unit tests for src/utils.py -- the signal-processing primitives.

Run with: python -m pytest tests/ -v   (from the project root)
"""

import numpy as np
import pytest

from src.utils import (
    bandpass_filter,
    bland_altman_stats,
    detrend,
    estimate_hr_fft,
    mae,
    pearson_corr,
    rmse,
)


def make_synthetic_pulse(bpm: float, fps: float, duration_sec: float, noise_std: float = 0.0,
                          trend_amp: float = 0.0, seed: int = 0) -> np.ndarray:
    """Build a clean synthetic pulse-like signal at a known BPM, with
    optional additive Gaussian noise and a slow linear trend (to test
    detrending/filtering)."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fps)
    t = np.arange(n) / fps
    freq_hz = bpm / 60.0
    # A few harmonics to look more like a real PPG waveform than a pure sine.
    sig = (
        1.0 * np.sin(2 * np.pi * freq_hz * t)
        + 0.3 * np.sin(2 * np.pi * 2 * freq_hz * t + 0.5)
        + 0.1 * np.sin(2 * np.pi * 3 * freq_hz * t + 1.0)
    )
    if trend_amp:
        sig = sig + trend_amp * t  # slow linear drift
    if noise_std:
        sig = sig + rng.normal(0, noise_std, size=n)
    return sig


class TestDetrend:
    def test_removes_linear_trend(self):
        fps = 30.0
        sig = make_synthetic_pulse(bpm=70, fps=fps, duration_sec=10, trend_amp=2.0)
        detrended = detrend(sig)
        # The detrended signal's own linear-fit slope should be near zero.
        t = np.arange(len(detrended))
        slope = np.polyfit(t, detrended, 1)[0]
        raw_slope = np.polyfit(t, sig, 1)[0]
        assert abs(slope) < abs(raw_slope) * 0.1

    def test_handles_short_signal(self):
        sig = np.array([1.0, 2.0])
        out = detrend(sig)
        assert out.shape == sig.shape

    def test_output_shape_matches_input(self):
        sig = np.random.randn(300)
        assert detrend(sig).shape == sig.shape


class TestBandpassFilter:
    def test_passes_hr_band_frequency(self):
        fps = 30.0
        bpm = 75.0
        sig = make_synthetic_pulse(bpm=bpm, fps=fps, duration_sec=15)
        filtered = bandpass_filter(sig, fps=fps)
        # Power should be concentrated at bpm/60 Hz.
        est = estimate_hr_fft(filtered, fps=fps)
        assert abs(est.bpm - bpm) < 2.0

    def test_attenuates_out_of_band_frequency(self):
        fps = 30.0
        n = 450
        t = np.arange(n) / fps
        # 0.1 Hz (6 BPM) is well below the 42 BPM lower bound.
        low_freq_sig = np.sin(2 * np.pi * 0.1 * t)
        filtered = bandpass_filter(low_freq_sig, fps=fps)
        assert np.std(filtered) < np.std(low_freq_sig) * 0.1

    def test_raises_on_nyquist_violation(self):
        with pytest.raises(ValueError):
            bandpass_filter(np.random.randn(100), fps=5.0)  # Nyquist=2.5Hz < high_hz=4Hz

    def test_raises_on_short_signal(self):
        with pytest.raises(ValueError):
            bandpass_filter(np.random.randn(5), fps=30.0)


class TestEstimateHrFft:
    @pytest.mark.parametrize("bpm", [45, 60, 75, 90, 120, 180])
    def test_recovers_known_bpm_clean_signal(self, bpm):
        fps = 30.0
        sig = make_synthetic_pulse(bpm=bpm, fps=fps, duration_sec=20)
        filtered = bandpass_filter(detrend(sig), fps=fps)
        est = estimate_hr_fft(filtered, fps=fps)
        assert abs(est.bpm - bpm) < 1.5, f"Expected ~{bpm} BPM, got {est.bpm}"
        assert est.confidence > 0.3

    def test_recovers_bpm_with_moderate_noise(self):
        fps = 30.0
        bpm = 80.0
        sig = make_synthetic_pulse(bpm=bpm, fps=fps, duration_sec=20, noise_std=0.5, seed=1)
        filtered = bandpass_filter(detrend(sig), fps=fps)
        est = estimate_hr_fft(filtered, fps=fps)
        assert abs(est.bpm - bpm) < 5.0

    def test_raises_on_empty_signal(self):
        with pytest.raises(ValueError):
            estimate_hr_fft(np.array([]), fps=30.0)

    def test_raises_on_constant_signal(self):
        with pytest.raises(ValueError):
            estimate_hr_fft(np.ones(300), fps=30.0)

    def test_confidence_lower_for_noisy_signal(self):
        fps = 30.0
        clean = make_synthetic_pulse(bpm=70, fps=fps, duration_sec=20)
        noisy = make_synthetic_pulse(bpm=70, fps=fps, duration_sec=20, noise_std=3.0, seed=2)
        clean_f = bandpass_filter(detrend(clean), fps=fps)
        noisy_f = bandpass_filter(detrend(noisy), fps=fps)
        est_clean = estimate_hr_fft(clean_f, fps=fps)
        est_noisy = estimate_hr_fft(noisy_f, fps=fps)
        assert est_clean.confidence >= est_noisy.confidence


class TestMetrics:
    def test_mae_zero_for_identical(self):
        x = np.array([70.0, 80.0, 90.0])
        assert mae(x, x) == 0.0

    def test_mae_known_value(self):
        pred = np.array([70.0, 80.0])
        target = np.array([75.0, 78.0])
        assert mae(pred, target) == pytest.approx((5.0 + 2.0) / 2)

    def test_rmse_geq_mae(self):
        pred = np.array([70.0, 80.0, 95.0])
        target = np.array([72.0, 79.0, 60.0])
        assert rmse(pred, target) >= mae(pred, target)

    def test_pearson_perfect_correlation(self):
        target = np.array([60.0, 70.0, 80.0, 90.0])
        pred = target * 1.0 + 2.0  # perfectly linearly related
        assert pearson_corr(pred, target) == pytest.approx(1.0, abs=1e-6)

    def test_pearson_zero_variance_returns_zero(self):
        pred = np.array([70.0, 70.0, 70.0])
        target = np.array([60.0, 70.0, 80.0])
        assert pearson_corr(pred, target) == 0.0

    def test_bland_altman_bias_sign(self):
        pred = np.array([72.0, 82.0, 92.0])
        target = np.array([70.0, 80.0, 90.0])
        stats = bland_altman_stats(pred, target)
        assert stats["bias"] == pytest.approx(2.0)
        assert stats["loa_lower"] <= stats["bias"] <= stats["loa_upper"]
