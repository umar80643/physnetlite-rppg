"""
Signal-processing and metrics utilities shared across the rPPG pipeline.

This module contains the "boring but critical" primitives: detrending,
bandpass filtering, FFT-based peak (heart-rate) extraction, and the
evaluation metrics used throughout the project. Bugs here are silent and
propagate into every downstream number, so every function is covered by
a unit test in tests/test_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal
from scipy.sparse import spdiags

# Physiologically plausible heart-rate band.
HR_BAND_HZ = (0.7, 4.0)  # 42-240 BPM
HR_BAND_BPM = (HR_BAND_HZ[0] * 60.0, HR_BAND_HZ[1] * 60.0)


def detrend(signal_1d: np.ndarray, lambda_: float = 100.0) -> np.ndarray:
    """Remove slow non-stationary trends using the smoothness-prior
    detrending method of Tarvainen et al. (2002), standard in HRV/rPPG
    literature. Falls back to simple linear detrend for very short signals.

    Args:
        signal_1d: 1D array of length N.
        lambda_: Smoothing parameter; larger = more low-frequency content
            removed. 100 is standard for a ~30 fps rPPG signal.

    Returns:
        Detrended signal, same shape as input.
    """
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    n = len(signal_1d)
    if n < 3:
        return signal_1d - np.mean(signal_1d)

    identity = np.eye(n)
    d2 = spdiags(
        np.array([np.ones(n), -2 * np.ones(n), np.ones(n)]),
        [0, 1, 2],
        n - 2,
        n,
    ).toarray()
    trend_matrix = identity - np.linalg.inv(identity + (lambda_**2) * d2.T @ d2)
    return trend_matrix @ signal_1d


def bandpass_filter(
    signal_1d: np.ndarray,
    fps: float,
    low_hz: float = HR_BAND_HZ[0],
    high_hz: float = HR_BAND_HZ[1],
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter restricted to the plausible
    heart-rate frequency band.

    Args:
        signal_1d: 1D input signal.
        fps: Sampling rate of `signal_1d` in Hz (i.e. video frame rate).
        low_hz: Lower cutoff frequency in Hz.
        high_hz: Upper cutoff frequency in Hz.
        order: Butterworth filter order.

    Returns:
        Filtered signal, same shape as input.

    Raises:
        ValueError: If fps is too low to resolve high_hz (Nyquist violated)
            or the signal is too short to filtify stably.
    """
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    nyquist = fps / 2.0
    if high_hz >= nyquist:
        raise ValueError(
            f"high_hz ({high_hz}) must be below Nyquist frequency "
            f"({nyquist} Hz for fps={fps}). Increase fps or lower high_hz."
        )
    if len(signal_1d) < 3 * order:
        raise ValueError(
            f"Signal too short ({len(signal_1d)} samples) to stably "
            f"filtfilt with order={order}. Need at least {3 * order} samples."
        )

    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, signal_1d)


@dataclass
class HREstimate:
    bpm: float
    confidence: float  # ratio of power at peak vs total power in-band, in [0, 1]
    freqs_hz: np.ndarray
    power: np.ndarray


def estimate_hr_fft(
    signal_1d: np.ndarray,
    fps: float,
    low_bpm: float = HR_BAND_BPM[0],
    high_bpm: float = HR_BAND_BPM[1],
    zero_pad_factor: int = 8,
) -> HREstimate:
    """Estimate heart rate (BPM) from a 1D pulse signal via FFT peak-picking.

    The signal is windowed (Hann), zero-padded for frequency resolution,
    and the dominant peak within [low_bpm, high_bpm] is taken as the HR.
    A confidence score (fraction of in-band spectral power concentrated
    at the peak's immediate neighborhood) is also returned so callers can
    flag unreliable estimates (e.g. during motion artifacts).

    Args:
        signal_1d: 1D pulse signal (ideally already bandpassed).
        fps: Sampling rate in Hz.
        low_bpm: Lower bound of the search range in BPM.
        high_bpm: Upper bound of the search range in BPM.
        zero_pad_factor: FFT length multiplier for finer frequency
            resolution (interpolation, not new information).

    Returns:
        HREstimate with the peak BPM, a confidence in [0, 1], and the
        full in-band spectrum for optional plotting.

    Raises:
        ValueError: If the signal is empty or has zero variance.
    """
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    if len(signal_1d) == 0:
        raise ValueError("Cannot estimate HR from an empty signal.")
    if np.std(signal_1d) < 1e-12:
        raise ValueError("Signal has ~zero variance; cannot estimate HR.")

    n = len(signal_1d)
    windowed = signal_1d * np.hanning(n)
    n_fft = int(2 ** np.ceil(np.log2(n * zero_pad_factor)))

    fft_vals = np.fft.rfft(windowed, n=n_fft)
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    power = np.abs(fft_vals) ** 2

    freqs_bpm = freqs_hz * 60.0
    in_band = (freqs_bpm >= low_bpm) & (freqs_bpm <= high_bpm)
    if not np.any(in_band):
        raise ValueError(
            f"No frequency bins in [{low_bpm}, {high_bpm}] BPM at fps={fps}. "
            "Signal too short or fps too low."
        )

    band_freqs = freqs_bpm[in_band]
    band_power = power[in_band]

    peak_idx = int(np.argmax(band_power))
    bpm = float(band_freqs[peak_idx])

    # Confidence: power within +-6 BPM of the peak, over total in-band power.
    near_peak = np.abs(band_freqs - bpm) <= 6.0
    confidence = float(band_power[near_peak].sum() / (band_power.sum() + 1e-12))

    return HREstimate(bpm=bpm, confidence=confidence, freqs_hz=band_freqs / 60.0, power=band_power)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average, used for smoothing BPM traces
    across sliding windows in the live demo."""
    if window <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #

def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error in BPM."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error in BPM."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient between predicted and ground-truth BPM."""
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if np.std(pred) < 1e-12 or np.std(target) < 1e-12:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def bland_altman_stats(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute Bland-Altman statistics: mean difference (bias) and the
    95% limits of agreement (bias +- 1.96 * std of differences).
    """
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    diffs = pred - target
    means = (pred + target) / 2.0
    bias = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return {
        "means": means,
        "diffs": diffs,
        "bias": bias,
        "loa_upper": bias + 1.96 * std_diff,
        "loa_lower": bias - 1.96 * std_diff,
        "std_diff": std_diff,
    }
