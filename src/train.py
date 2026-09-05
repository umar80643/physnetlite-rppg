"""
Stage 3: Training script for PhysNetLite.

Trains on windows from the train split (subject-level), validates on the
val split each epoch, and saves the best checkpoint (by val MAE in BPM,
computed via FFT on the predicted waveform) to results/checkpoints/.

Usage:
    python -m src.train --data-root data/raw --epochs 30 --batch-size 4

Run `python -m src.train --help` for all options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import RPPGWindowDataset, discover_subjects, subject_level_split
from src.model import NegativePearsonLoss, PhysNetLite
from src.utils import bandpass_filter, detrend, estimate_hr_fft, mae


def waveform_to_bpm_batch(waveforms: np.ndarray, fps: float) -> np.ndarray:
    """Convert a batch of predicted waveforms (B, T) to BPM estimates via
    the same FFT peak-picking used everywhere else in the project, so
    training-time validation numbers are directly comparable to the
    final evaluate.py results."""
    bpms = []
    for w in waveforms:
        try:
            filtered = bandpass_filter(detrend(w), fps=fps)
            bpms.append(estimate_hr_fft(filtered, fps=fps).bpm)
        except ValueError:
            bpms.append(np.nan)
    return np.array(bpms)


def run_epoch(model, loader, loss_fn, optimizer, device, fps, train: bool) -> dict:
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    all_pred_bpm, all_target_bpm = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for clip, target_bpm in loader:
            clip = clip.to(device)
            target_bpm = target_bpm.to(device)

            # We only have a scalar target BPM per clip (not a full
            # ground-truth waveform, which UBFC-rPPG/PURE do provide but
            # which we've simplified to per-frame HR in dataset.py for a
            # smaller/simpler pipeline). To still use the Pearson loss's
            # scale-invariance benefits, we synthesize a target sinusoid
            # at the ground-truth BPM and train the model to correlate
            # with it -- a standard simplification when only HR labels
            # (not raw waveforms) are used. See README for the note on
            # using the full waveform instead for a stronger baseline.
            t_len = clip.shape[2]
            t_axis = torch.arange(t_len, device=device).float() / fps
            freq_hz = (target_bpm / 60.0).unsqueeze(1)
            target_waveform = torch.sin(2 * np.pi * freq_hz * t_axis.unsqueeze(0))

            pred_waveform = model(clip)
            loss = loss_fn(pred_waveform, target_waveform)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pred_bpm = waveform_to_bpm_batch(pred_waveform.detach().cpu().numpy(), fps)
            all_pred_bpm.append(pred_bpm)
            all_target_bpm.append(target_bpm.cpu().numpy())

    all_pred_bpm = np.concatenate(all_pred_bpm) if all_pred_bpm else np.array([])
    all_target_bpm = np.concatenate(all_target_bpm) if all_target_bpm else np.array([])
    valid = ~np.isnan(all_pred_bpm)

    return {
        "loss": total_loss / max(1, n_batches),
        "mae_bpm": mae(all_pred_bpm[valid], all_target_bpm[valid]) if valid.any() else float("nan"),
        "n_valid": int(valid.sum()),
        "n_total": len(all_pred_bpm),
    }


def main():
    parser = argparse.ArgumentParser(description="Train PhysNetLite for rPPG BPM estimation.")
    parser.add_argument("--data-root", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-len", type=int, default=150, help="Frames per clip (5s @ 30fps).")
    parser.add_argument("--stride", type=int, default=30, help="Frame hop between windows.")
    parser.add_argument("--roi-size", type=int, default=72, help="Square ROI crop size in pixels.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    subjects = discover_subjects(args.data_root)
    print(f"Discovered {len(subjects)} subjects across UBFC-rPPG/PURE.")
    splits = subject_level_split(subjects, seed=args.seed)
    for name, subs in splits.items():
        print(f"  {name}: {len(subs)} subjects -> {[s.subject_id for s in subs]}")

    common_kwargs = dict(
        clip_len_frames=args.clip_len,
        stride_frames=args.stride,
        roi_size=(args.roi_size, args.roi_size),
    )
    train_ds = RPPGWindowDataset(splits["train"], **common_kwargs)
    val_ds = RPPGWindowDataset(splits["val"], **common_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = PhysNetLite(base_channels=args.base_channels).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    loss_fn = NegativePearsonLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, loss_fn, optimizer, device, args.fps, train=True)
        val_stats = run_epoch(model, val_loader, loss_fn, optimizer, device, args.fps, train=False)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_stats['loss']:.4f} MAE {train_stats['mae_bpm']:.2f} | "
            f"val loss {val_stats['loss']:.4f} MAE {val_stats['mae_bpm']:.2f}"
        )
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})

        if val_stats["mae_bpm"] < best_val_mae:
            best_val_mae = val_stats["mae_bpm"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  -> New best val MAE {best_val_mae:.2f}, checkpoint saved.")

    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nTraining complete. Best val MAE: {best_val_mae:.2f} BPM.")
    print(f"Best checkpoint: {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
