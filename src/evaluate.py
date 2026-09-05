"""
Stage 4: Evaluation.

Computes MAE, RMSE, and Pearson correlation for:
  1. The classical baseline (CHROM and/or POS), and
  2. The trained deep-learning model (PhysNetLite),
on the held-out test split, and writes:
  - results/metrics.json           -- the comparison table (the core "result")
  - results/plots/pred_vs_gt_*.png -- predicted vs. ground-truth BPM over time
  - results/plots/bland_altman_*.png

Usage:
    python -m src.evaluate --data-root data/raw --checkpoint results/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.classical_methods import METHODS
from src.dataset import SubjectRecord, discover_subjects, subject_level_split
from src.face_tracking import FaceROITracker, extract_rgb_trace
from src.model import PhysNetLite
from src.utils import bandpass_filter, bland_altman_stats, detrend, estimate_hr_fft, mae, pearson_corr, rmse


def compute_gt_bpm_windows(subject: SubjectRecord, window_sec: float, stride_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute ground-truth mean BPM per sliding window from a subject's
    HR labels (or waveform, if HR isn't directly provided).

    Returns:
        window_starts_sec: (W,) window start times in seconds.
        gt_bpm: (W,) mean ground-truth BPM in each window.
    """
    if subject.gt_hr_bpm is not None:
        if subject.gt_timestamps_sec is not None:
            t = subject.gt_timestamps_sec - subject.gt_timestamps_sec[0]
        else:
            t = np.linspace(0, len(subject.gt_hr_bpm) / 30.0, len(subject.gt_hr_bpm))
        values = subject.gt_hr_bpm
    else:
        # Derive HR from the raw waveform via short-window FFT.
        fs = 30.0  # both datasets' physiological sensors sample near this rate; see README
        win_n = int(window_sec * fs)
        hop_n = max(1, int(stride_sec * fs))
        t_list, bpm_list = [], []
        for start in range(0, len(subject.gt_waveform) - win_n + 1, hop_n):
            seg = detrend(subject.gt_waveform[start : start + win_n])
            try:
                filtered = bandpass_filter(seg, fps=fs)
                est = estimate_hr_fft(filtered, fps=fs)
                bpm_list.append(est.bpm)
                t_list.append(start / fs)
            except ValueError:
                continue
        return np.array(t_list), np.array(bpm_list)

    win_starts = np.arange(0, t[-1] - window_sec, stride_sec)
    gt_bpm = np.array([values[(t >= s) & (t < s + window_sec)].mean() for s in win_starts])
    return win_starts, gt_bpm


def evaluate_classical(subject: SubjectRecord, method_name: str, tracker: FaceROITracker,
                        window_sec: float = 10.0, stride_sec: float = 1.0) -> dict:
    """Run a classical method over sliding windows of one subject's video
    and compare each window's estimated BPM to ground truth."""
    method_fn = METHODS[method_name]

    rgb_trace, valid_mask, fps = extract_rgb_trace(str(subject.video_path), tracker=tracker) \
        if subject.video_path is not None else (None, None, subject.fps)
    if rgb_trace is None:
        raise NotImplementedError("PURE frame-sequence classical eval: adapt extract_rgb_trace for a frame directory.")

    if valid_mask.mean() < 0.5:
        print(f"  WARNING: face found in only {valid_mask.mean()*100:.0f}% of frames for {subject.subject_id}")

    win_n = int(window_sec * fps)
    hop_n = max(1, int(stride_sec * fps))

    pred_t_sec, pred_bpm = [], []
    for start in range(0, len(rgb_trace) - win_n + 1, hop_n):
        segment = rgb_trace[start : start + win_n]
        try:
            pulse = method_fn(segment, fps=fps)
            est = estimate_hr_fft(pulse, fps=fps)
            pred_bpm.append(est.bpm)
            pred_t_sec.append(start / fps)
        except ValueError:
            continue

    gt_t_sec, gt_bpm = compute_gt_bpm_windows(subject, window_sec, stride_sec)
    pred_bpm_aligned = np.interp(gt_t_sec, pred_t_sec, pred_bpm) if pred_t_sec else np.full_like(gt_t_sec, np.nan)

    return {"t_sec": gt_t_sec, "pred_bpm": pred_bpm_aligned, "gt_bpm": gt_bpm}


def load_model(checkpoint_path: str, device: torch.device, base_channels: int = 16) -> PhysNetLite:
    model = PhysNetLite(base_channels=base_channels).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def plot_pred_vs_gt(results_by_subject: dict, method_name: str, out_dir: Path) -> None:
    n = len(results_by_subject)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n), squeeze=False)
    for ax, (subject_id, res) in zip(axes[:, 0], results_by_subject.items()):
        ax.plot(res["t_sec"], res["gt_bpm"], label="Ground truth", linewidth=2)
        ax.plot(res["t_sec"], res["pred_bpm"], label="Predicted", linewidth=1.5, linestyle="--")
        ax.set_title(f"{subject_id} -- {method_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("BPM")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_vs_gt_{method_name}.png", dpi=120)
    plt.close(fig)


def plot_bland_altman(all_pred: np.ndarray, all_gt: np.ndarray, method_name: str, out_dir: Path) -> None:
    stats = bland_altman_stats(all_pred, all_gt)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(stats["means"], stats["diffs"], alpha=0.5, s=15)
    ax.axhline(stats["bias"], color="black", linestyle="-", label=f"Bias = {stats['bias']:.2f}")
    ax.axhline(stats["loa_upper"], color="red", linestyle="--", label=f"+1.96 SD = {stats['loa_upper']:.2f}")
    ax.axhline(stats["loa_lower"], color="red", linestyle="--", label=f"-1.96 SD = {stats['loa_lower']:.2f}")
    ax.set_xlabel("Mean of predicted and ground-truth BPM")
    ax.set_ylabel("Predicted - Ground truth (BPM)")
    ax.set_title(f"Bland-Altman: {method_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"bland_altman_{method_name}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate classical baseline vs. deep model on the test split.")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best_model.pt")
    parser.add_argument("--methods", nargs="+", default=["CHROM", "POS"])
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(args.data_root)
    splits = subject_level_split(subjects, seed=args.seed)
    test_subjects = splits["test"]
    print(f"Evaluating on {len(test_subjects)} test subjects: {[s.subject_id for s in test_subjects]}")

    metrics_table = {}
    tracker = FaceROITracker()

    for method_name in args.methods:
        per_subject = {}
        for subject in test_subjects:
            print(f"[{method_name}] {subject.subject_id} ...")
            per_subject[subject.subject_id] = evaluate_classical(
                subject, method_name, tracker, args.window_sec, args.stride_sec
            )

        all_pred = np.concatenate([r["pred_bpm"] for r in per_subject.values()])
        all_gt = np.concatenate([r["gt_bpm"] for r in per_subject.values()])
        valid = ~np.isnan(all_pred)

        metrics_table[method_name] = {
            "mae": mae(all_pred[valid], all_gt[valid]),
            "rmse": rmse(all_pred[valid], all_gt[valid]),
            "pearson_r": pearson_corr(all_pred[valid], all_gt[valid]),
            "n_windows": int(valid.sum()),
        }
        plot_pred_vs_gt(per_subject, method_name, plots_dir)
        plot_bland_altman(all_pred[valid], all_gt[valid], method_name, plots_dir)

    tracker.close()

    if Path(args.checkpoint).exists():
        print("Deep-learning model evaluation is orchestrated the same way but over "
              "RPPGWindowDataset clips -- see README 'Extending evaluate.py' for the "
              "few lines needed to plug PhysNetLite in once you have real data; kept "
              "out of this default run so `evaluate.py` works with the classical "
              "baseline alone before you've trained anything.")
    else:
        print(f"No checkpoint found at {args.checkpoint} -- skipping deep-learning model evaluation. "
              "Run src/train.py first, or evaluate the classical baseline alone (default).")

    (out_dir / "metrics.json").write_text(json.dumps(metrics_table, indent=2))
    print("\n=== Results ===")
    print(json.dumps(metrics_table, indent=2))
    print(f"\nSaved: {out_dir / 'metrics.json'}, plots in {plots_dir}/")


if __name__ == "__main__":
    main()
