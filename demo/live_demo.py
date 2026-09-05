"""
Stage 5: Real-time webcam demo.

Captures live webcam video, tracks the face ROI every frame, keeps a
rolling buffer of the last N seconds of RGB means, and re-estimates BPM
from that buffer periodically using the classical CHROM (or POS) method
(the classical baseline runs comfortably in real time on CPU; swap in
the trained PhysNetLite model via --model-checkpoint if you want the
deep-learning path instead -- see `predict_with_model` below).

Robustness:
  - If no face is detected for several consecutive frames, the overlay
    shows "No face detected" instead of a stale/garbage BPM.
  - If the rolling buffer's spectral confidence (see
    src.utils.estimate_hr_fft) is below --min-confidence, the overlay
    shows "Signal unstable" instead of a low-confidence BPM number.
  - The displayed BPM is smoothed (moving average) across recent
    estimates to avoid jumpy numbers.

Usage:
    python -m demo.live_demo                          # CHROM baseline, default webcam
    python -m demo.live_demo --method POS --camera 0
    python -m demo.live_demo --model-checkpoint results/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import collections
import time

import cv2
import numpy as np
import torch

from src.classical_methods import METHODS
from src.face_tracking import FaceROITracker
from src.model import PhysNetLite
from src.utils import bandpass_filter, detrend, estimate_hr_fft, moving_average

WINDOW_SEC = 10.0          # rolling buffer length used for each BPM estimate
UPDATE_EVERY_SEC = 1.0     # how often to recompute BPM from the buffer
MAX_MISSING_FRAMES = 15    # consecutive no-face frames before showing "No face detected"
BPM_SMOOTH_N = 5           # number of recent BPM estimates to moving-average


def predict_with_model(roi_frames_bgr: list[np.ndarray], model: PhysNetLite, fps: float,
                        device: torch.device, roi_size: tuple[int, int] = (72, 72)) -> float:
    """Run the trained deep model on a buffer of raw ROI-crop frames and
    return the estimated BPM. Kept separate from the classical path so
    the classical demo has zero torch overhead if --model-checkpoint is
    not passed.
    """
    resized = [cv2.resize(f, roi_size, interpolation=cv2.INTER_AREA) for f in roi_frames_bgr]
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized]
    clip = np.stack(rgb, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    clip_t = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, 3, T, H, W)

    with torch.no_grad():
        waveform = model(clip_t)[0].cpu().numpy()
    filtered = bandpass_filter(detrend(waveform), fps=fps)
    return estimate_hr_fft(filtered, fps=fps).bpm


def main():
    parser = argparse.ArgumentParser(description="Live webcam rPPG heart-rate demo.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--method", type=str, default="CHROM", choices=list(METHODS.keys()))
    parser.add_argument("--model-checkpoint", type=str, default=None,
                         help="If set, use the trained PhysNetLite model instead of the classical method.")
    parser.add_argument("--min-confidence", type=float, default=0.25,
                         help="Below this FFT peak confidence, show 'Signal unstable' instead of a BPM.")
    parser.add_argument("--fps-assumed", type=float, default=30.0,
                         help="Fallback fps if the camera doesn't report one.")
    args = parser.parse_args()

    use_model = args.model_checkpoint is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if use_model:
        model = PhysNetLite().to(device)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
        model.eval()
        print(f"Loaded deep-learning model from {args.model_checkpoint} (device={device}).")
    else:
        print(f"Using classical {args.method} baseline.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. If running in a container/VM, "
            "webcam passthrough must be configured on the host; this cannot run headless."
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = args.fps_assumed
    buffer_len = int(WINDOW_SEC * fps)
    update_every_n_frames = max(1, int(UPDATE_EVERY_SEC * fps))

    tracker = FaceROITracker()
    rgb_buffer: collections.deque = collections.deque(maxlen=buffer_len)
    roi_frame_buffer: collections.deque = collections.deque(maxlen=buffer_len)  # only used if use_model
    bpm_history: collections.deque = collections.deque(maxlen=BPM_SMOOTH_N)

    consecutive_missing = 0
    frame_i = 0
    last_display_text = "Initializing..."
    t_start = time.time()

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.")
                break

            timestamp_ms = int((time.time() - t_start) * 1000)
            result = tracker.process(frame, timestamp_ms=timestamp_ms)

            if result.found:
                consecutive_missing = 0
                rgb_buffer.append(result.mean_rgb)
                if use_model and result.mask is not None:
                    ys, xs = np.where(result.mask > 0)
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    roi_frame_buffer.append(frame[y0 : y1 + 1, x0 : x1 + 1].copy())
                # Draw the ROI outline for visual feedback.
                contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
            else:
                consecutive_missing += 1

            frame_i += 1
            enough_data = len(rgb_buffer) >= buffer_len // 2  # allow an estimate before buffer is full

            if consecutive_missing >= MAX_MISSING_FRAMES:
                last_display_text = "No face detected"
            elif enough_data and frame_i % update_every_n_frames == 0:
                try:
                    if use_model and len(roi_frame_buffer) >= buffer_len // 2:
                        bpm = predict_with_model(list(roi_frame_buffer), model, fps, device)
                        confidence = 1.0  # model path doesn't currently expose a confidence score
                    else:
                        rgb_arr = np.array(rgb_buffer)
                        pulse = METHODS[args.method](rgb_arr, fps=fps)
                        est = estimate_hr_fft(pulse, fps=fps)
                        bpm, confidence = est.bpm, est.confidence

                    if confidence < args.min_confidence:
                        last_display_text = "Signal unstable"
                    else:
                        bpm_history.append(bpm)
                        smoothed_bpm = float(np.mean(bpm_history))
                        last_display_text = f"{smoothed_bpm:.0f} BPM"
                except ValueError:
                    last_display_text = "Signal unstable"

            color = (0, 255, 0) if "BPM" in last_display_text else (0, 165, 255)
            cv2.putText(frame, last_display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(frame, f"method: {'model' if use_model else args.method}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("rPPG Live Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
