"""
Synthetic end-to-end pipeline smoke test.

This is NOT a substitute for evaluating on real UBFC-rPPG/PURE data --
it exists to prove, without any dataset or webcam, that every stage of
the pipeline (signal processing -> classical baseline -> model ->
training loop -> metrics -> plots) is wired together correctly and
produces sane numbers, end to end. It bypasses face detection entirely
(no real face image is available in this environment) and instead
synthesizes per-frame RGB traces and matching video-like tensors
directly, at several known ground-truth BPMs and noise levels.

Run: python notebooks/synthetic_pipeline_smoke_test.py
Outputs: results/plots/synthetic_*.png, prints a metrics summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.classical_methods import chrom, pos
from src.model import NegativePearsonLoss, PhysNetLite
from src.utils import bandpass_filter, bland_altman_stats, detrend, estimate_hr_fft, mae, pearson_corr, rmse

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def synth_rgb_trace(bpm: float, fps: float, duration_sec: float, noise_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fps)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t) + 0.25 * np.sin(2 * np.pi * 2 * (bpm / 60.0) * t + 0.3)
    base_r, base_g, base_b = 150.0, 100.0, 80.0
    r = base_r + 0.6 * pulse + rng.normal(0, noise_std, n)
    g = base_g + 1.0 * pulse + rng.normal(0, noise_std, n)
    b = base_b + 0.3 * pulse + rng.normal(0, noise_std, n)
    drift = 4.0 * np.sin(2 * np.pi * 0.04 * t)
    return np.stack([r + drift, g + drift, b + drift], axis=1)


def synth_video_clip(bpm: float, fps: float, n_frames: int, hw: int, seed: int) -> torch.Tensor:
    """Synthesize a (3, T, H, W) tensor whose mean pixel intensity per
    frame carries a sinusoidal modulation at `bpm`, so PhysNetLite has a
    genuine (if simplified) spatiotemporal signal to learn from, rather
    than pure noise. Standard trick for smoke-testing spatiotemporal
    models without real video."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)  # (T,)
    base = rng.uniform(0.3, 0.6, size=(3, 1, hw, hw)).astype(np.float32)
    modulation = 0.05 * pulse.reshape(1, n_frames, 1, 1).astype(np.float32)
    noise = rng.normal(0, 0.02, size=(3, n_frames, hw, hw)).astype(np.float32)
    clip = base + modulation + noise
    return torch.from_numpy(np.clip(clip, 0, 1))


def run_classical_baseline_check():
    print("=" * 70)
    print("STAGE 2 CHECK: Classical CHROM/POS baseline on synthetic RGB traces")
    print("=" * 70)
    fps = 30.0
    test_bpms = np.arange(50, 150, 5)
    results = {"CHROM": {"pred": [], "gt": []}, "POS": {"pred": [], "gt": []}}

    for i, bpm in enumerate(test_bpms):
        rgb = synth_rgb_trace(bpm, fps, duration_sec=15, noise_std=0.4, seed=i)
        for name, fn in (("CHROM", chrom), ("POS", pos)):
            pulse = fn(rgb, fps=fps)
            est = estimate_hr_fft(pulse, fps=fps)
            results[name]["pred"].append(est.bpm)
            results[name]["gt"].append(bpm)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, results.keys()):
        pred, gt = np.array(results[name]["pred"]), np.array(results[name]["gt"])
        m, r, p = mae(pred, gt), rmse(pred, gt), pearson_corr(pred, gt)
        print(f"  {name:6s}  MAE={m:5.2f} BPM  RMSE={r:5.2f} BPM  Pearson r={p:.3f}")
        ax.scatter(gt, pred, alpha=0.7)
        ax.plot([gt.min(), gt.max()], [gt.min(), gt.max()], "k--", linewidth=1, label="Ideal")
        ax.set_xlabel("Ground-truth BPM (synthetic)")
        ax.set_ylabel("Estimated BPM")
        ax.set_title(f"{name}: MAE={m:.2f}, r={p:.3f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "synthetic_classical_baseline.png", dpi=120)
    plt.close(fig)
    print(f"  Saved plot: {PLOTS_DIR / 'synthetic_classical_baseline.png'}")

    # Bland-Altman for CHROM as an example.
    pred, gt = np.array(results["CHROM"]["pred"]), np.array(results["CHROM"]["gt"])
    stats = bland_altman_stats(pred, gt)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(stats["means"], stats["diffs"], alpha=0.7)
    ax.axhline(stats["bias"], color="black", label=f"Bias={stats['bias']:.2f}")
    ax.axhline(stats["loa_upper"], color="red", linestyle="--")
    ax.axhline(stats["loa_lower"], color="red", linestyle="--")
    ax.set_xlabel("Mean of predicted & GT BPM")
    ax.set_ylabel("Predicted - GT (BPM)")
    ax.set_title("Bland-Altman: CHROM (synthetic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "synthetic_bland_altman_chrom.png", dpi=120)
    plt.close(fig)
    print(f"  Saved plot: {PLOTS_DIR / 'synthetic_bland_altman_chrom.png'}")
    return results


def run_model_learns_check():
    print("\n" + "=" * 70)
    print("STAGE 3 CHECK: Does PhysNetLite's loss go down on synthetic clips?")
    print("=" * 70)
    torch.manual_seed(0)
    fps = 30.0
    n_frames = 60
    hw = 36
    model = PhysNetLite(base_channels=8)
    loss_fn = NegativePearsonLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    bpms = [65, 75, 85, 95]
    clips = torch.stack([synth_video_clip(b, fps, n_frames, hw, seed=i) for i, b in enumerate(bpms)])
    t = np.arange(n_frames) / fps
    targets = torch.stack(
        [torch.from_numpy(np.sin(2 * np.pi * (b / 60.0) * t)).float() for b in bpms]
    )

    losses = []
    for step in range(60):
        optimizer.zero_grad()
        pred = model(clips)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == 59:
            print(f"  step {step:3d}  loss={loss.item():.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Negative Pearson loss")
    ax.set_title("PhysNetLite loss on synthetic clips (sanity check)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "synthetic_training_loss.png", dpi=120)
    plt.close(fig)
    print(f"  Saved plot: {PLOTS_DIR / 'synthetic_training_loss.png'}")

    assert losses[-1] < losses[0], "Loss did not decrease -- something is broken in the training loop."
    print(f"  PASS: loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    # Check the model's predicted BPM (via FFT) after training on these
    # 4 clips -- should land closer to their true BPMs than at init.
    model.eval()
    with torch.no_grad():
        pred_waveforms = model(clips).numpy()
    print("\n  Post-training BPM recovery on the 4 training clips (expect rough alignment,")
    print("  NOT held-out generalization -- this model has only seen these 4 clips):")
    for b, w in zip(bpms, pred_waveforms):
        filtered = bandpass_filter(detrend(w), fps=fps)
        try:
            est_bpm = estimate_hr_fft(filtered, fps=fps).bpm
            print(f"    true={b:3d} BPM  ->  predicted={est_bpm:6.1f} BPM")
        except ValueError:
            print(f"    true={b:3d} BPM  ->  predicted=<no stable peak>")


if __name__ == "__main__":
    run_classical_baseline_check()
    run_model_learns_check()
    print("\nAll synthetic smoke checks completed. See results/plots/synthetic_*.png")
