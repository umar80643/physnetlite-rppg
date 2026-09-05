"""
Dataset loading for UBFC-rPPG and PURE.

Both datasets are organized as one folder per subject, containing a video
(or frame sequence) and a ground-truth waveform (or per-frame HR) recorded
with a contact pulse oximeter. This module normalizes both into a common
`SubjectRecord` representation and exposes:

  - `discover_subjects`: scan a raw dataset directory into SubjectRecords.
  - `subject_level_split`: split subjects (not frames!) into train/val/test.
  - `RPPGWindowDataset`: a PyTorch Dataset that yields
        (clip: (T, C, H, W) float tensor, target_bpm: float)
    for fixed-length sliding windows, built from the face ROI crops.

Splitting by subject (not by frame) is essential: consecutive frames from
the same subject are highly correlated (same skin tone, lighting, camera),
so a frame-level split would leak subject identity between train and test
and produce optimistic, meaningless metrics.

Expected raw layout (see README "Dataset setup" for exact download
instructions):

    data/raw/UBFC-rPPG/
        subject1/
            vid.avi
            ground_truth.txt      # line 1: PPG waveform, line 2: HR per frame, line 3: timestamps
        subject3/
            ...

    data/raw/PURE/
        01-01/
            01-01/                # folder of PNG frames
            01-01.json            # contains pulse waveform + per-frame HR
        ...

Both datasets require filling out a short academic-use request form with
the original authors before download -- this module does not (and cannot)
fetch them automatically. See README.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for dataset.py. Install with `pip install torch`.") from exc

import cv2

from src.face_tracking import FaceROITracker, build_roi_mask


@dataclass
class SubjectRecord:
    """Normalized representation of one subject's recording, regardless
    of which source dataset (UBFC-rPPG or PURE) it came from."""
    subject_id: str
    dataset: str  # "UBFC-rPPG" or "PURE"
    video_path: Path | None       # for UBFC-rPPG (single video file)
    frames_dir: Path | None       # for PURE (directory of PNG frames)
    fps: float
    gt_waveform: np.ndarray       # (N_gt,) raw ground-truth PPG waveform
    gt_hr_bpm: np.ndarray | None  # (N_gt,) per-sample HR in BPM, if provided directly
    gt_timestamps_sec: np.ndarray | None  # (N_gt,) timestamps, if provided


def _parse_ubfc_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Parse UBFC-rPPG's `ground_truth.txt`: line 1 = PPG waveform,
    line 2 = per-frame HR (BPM), line 3 = timestamps (seconds)."""
    lines = path.read_text().strip().splitlines()
    waveform = np.array([float(x) for x in lines[0].split()])
    hr = np.array([float(x) for x in lines[1].split()]) if len(lines) > 1 else None
    timestamps = np.array([float(x) for x in lines[2].split()]) if len(lines) > 2 else None
    return waveform, hr, timestamps


def _parse_pure_json(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Parse PURE's per-subject JSON file (`/FullPackage` field: list of
    {Timestamp, Value: {pulseRate, waveform, ...}} entries)."""
    data = json.loads(path.read_text())
    entries = data.get("/FullPackage", data.get("FullPackage", []))
    waveform = np.array([e["Value"]["waveform"] for e in entries], dtype=np.float64)
    hr = np.array([e["Value"]["pulseRate"] for e in entries], dtype=np.float64)
    timestamps = np.array([e["Timestamp"] for e in entries], dtype=np.float64) / 1e9  # ns -> s
    return waveform, hr, timestamps


def discover_subjects(raw_root: str | Path) -> list[SubjectRecord]:
    """Scan `data/raw/UBFC-rPPG` and/or `data/raw/PURE` and build a list
    of SubjectRecords. Missing datasets are silently skipped (so you can
    develop/test with just one of the two).

    Args:
        raw_root: Path to the `data/raw` directory.

    Returns:
        List of SubjectRecord, one per discovered subject.
    """
    raw_root = Path(raw_root)
    subjects: list[SubjectRecord] = []

    ubfc_root = raw_root / "UBFC-rPPG"
    if ubfc_root.exists():
        for subject_dir in sorted(ubfc_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            video_candidates = list(subject_dir.glob("*.avi")) + list(subject_dir.glob("*.mp4"))
            gt_path = subject_dir / "ground_truth.txt"
            if not video_candidates or not gt_path.exists():
                continue
            waveform, hr, ts = _parse_ubfc_ground_truth(gt_path)
            cap = cv2.VideoCapture(str(video_candidates[0]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            subjects.append(
                SubjectRecord(
                    subject_id=subject_dir.name,
                    dataset="UBFC-rPPG",
                    video_path=video_candidates[0],
                    frames_dir=None,
                    fps=fps,
                    gt_waveform=waveform,
                    gt_hr_bpm=hr,
                    gt_timestamps_sec=ts,
                )
            )

    pure_root = raw_root / "PURE"
    if pure_root.exists():
        for subject_dir in sorted(pure_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            json_candidates = list(subject_dir.glob("*.json"))
            frames_candidates = [d for d in subject_dir.iterdir() if d.is_dir()]
            if not json_candidates or not frames_candidates:
                continue
            waveform, hr, ts = _parse_pure_json(json_candidates[0])
            # PURE is recorded at 30fps by protocol.
            subjects.append(
                SubjectRecord(
                    subject_id=subject_dir.name,
                    dataset="PURE",
                    video_path=None,
                    frames_dir=frames_candidates[0],
                    fps=30.0,
                    gt_waveform=waveform,
                    gt_hr_bpm=hr,
                    gt_timestamps_sec=ts,
                )
            )

    return subjects


def subject_level_split(
    subjects: list[SubjectRecord],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[SubjectRecord]]:
    """Split subjects (not frames) into train/val/test.

    Splitting at the subject level prevents identity leakage: frames
    from the same person in train and test would let a model "cheat" by
    memorizing skin tone / lighting / camera characteristics rather than
    learning the actual pulse signal, producing metrics that look good
    but don't generalize.

    Args:
        subjects: All discovered subjects.
        train_frac: Fraction of subjects for training.
        val_frac: Fraction of subjects for validation (test gets the rest).
        seed: RNG seed for reproducible splits.

    Returns:
        Dict with keys "train", "val", "test", each a list of SubjectRecord.
    """
    if not subjects:
        raise ValueError("No subjects found -- check that data/raw/UBFC-rPPG or data/raw/PURE exists and is populated.")

    rng = np.random.default_rng(seed)
    ids = [s.subject_id for s in subjects]
    order = rng.permutation(len(ids))

    n_train = max(1, int(round(train_frac * len(ids))))
    n_val = max(1, int(round(val_frac * len(ids)))) if len(ids) > 2 else 0

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    by_idx = lambda idxs: [subjects[i] for i in idxs]  # noqa: E731
    return {"train": by_idx(train_idx), "val": by_idx(val_idx), "test": by_idx(test_idx)}


class RPPGWindowDataset(Dataset):
    """PyTorch Dataset yielding fixed-length face-ROI clips paired with a
    single target BPM (the mean ground-truth HR over the clip window).

    Face detection + ROI cropping is done lazily and cached per-subject
    the first time that subject's frames are requested, to avoid re-running
    MediaPipe on every __getitem__ call.
    """

    def __init__(
        self,
        subjects: list[SubjectRecord],
        clip_len_frames: int = 150,  # 5s @ 30fps
        stride_frames: int = 30,     # 1s hop between windows
        roi_size: tuple[int, int] = (72, 72),
        tracker: FaceROITracker | None = None,
    ):
        """
        Args:
            subjects: Subjects to draw windows from (one split's list).
            clip_len_frames: Number of frames per training clip.
            stride_frames: Hop size between consecutive windows (smaller
                = more overlap = more training windows per video).
            roi_size: Output (H, W) each cropped ROI frame is resized to.
            tracker: Shared FaceROITracker instance. If None, one is
                created lazily on first use (requires the FaceLandmarker
                model -- see face_tracking.py).
        """
        self.subjects = subjects
        self.clip_len_frames = clip_len_frames
        self.stride_frames = stride_frames
        self.roi_size = roi_size
        self._tracker = tracker
        self._index: list[tuple[int, int]] = []  # (subject_idx, start_frame)
        self._frame_cache: dict[int, np.ndarray] = {}  # subject_idx -> (N, H, W, 3) ROI crops
        self._hr_cache: dict[int, np.ndarray] = {}      # subject_idx -> (N,) per-frame BPM

        self._build_index()

    def _get_tracker(self) -> FaceROITracker:
        if self._tracker is None:
            self._tracker = FaceROITracker()
        return self._tracker

    def _load_subject_frames_and_hr(self, subject_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if subject_idx in self._frame_cache:
            return self._frame_cache[subject_idx], self._hr_cache[subject_idx]

        subject = self.subjects[subject_idx]
        tracker = self._get_tracker()
        tracker.reset()

        roi_frames = []
        if subject.video_path is not None:
            cap = cv2.VideoCapture(str(subject.video_path))
            frame_i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = int(frame_i * (1000.0 / subject.fps))
                result = tracker.process(frame, timestamp_ms=ts_ms)
                roi_frames.append(self._crop_roi(frame, result))
                frame_i += 1
            cap.release()
        else:
            frame_paths = sorted(subject.frames_dir.glob("*.png"), key=lambda p: p.name)
            for frame_i, fp in enumerate(frame_paths):
                frame = cv2.imread(str(fp))
                ts_ms = int(frame_i * (1000.0 / subject.fps))
                result = tracker.process(frame, timestamp_ms=ts_ms)
                roi_frames.append(self._crop_roi(frame, result))

        roi_array = np.stack(roi_frames, axis=0)  # (N, H, W, 3) uint8

        # Per-frame HR ground truth: use provided per-frame HR if
        # available, else derive from the waveform's local instantaneous
        # rate is out of scope here -- we resample the coarser HR/waveform
        # onto the video's frame count via linear interpolation on time.
        n_frames = roi_array.shape[0]
        video_t = np.arange(n_frames) / subject.fps

        if subject.gt_hr_bpm is not None and subject.gt_timestamps_sec is not None:
            hr_t = subject.gt_timestamps_sec - subject.gt_timestamps_sec[0]
            hr_vals = subject.gt_hr_bpm
        elif subject.gt_hr_bpm is not None:
            hr_t = np.linspace(0, video_t[-1], len(subject.gt_hr_bpm))
            hr_vals = subject.gt_hr_bpm
        else:
            raise ValueError(
                f"Subject {subject.subject_id} has no per-frame HR ground truth; "
                "derive it from the waveform first (see evaluate.py compute_gt_bpm_from_waveform)."
            )

        per_frame_hr = np.interp(video_t, hr_t, hr_vals)

        self._frame_cache[subject_idx] = roi_array
        self._hr_cache[subject_idx] = per_frame_hr
        return roi_array, per_frame_hr

    def _crop_roi(self, frame_bgr: np.ndarray, result) -> np.ndarray:
        """Crop the bounding box of the ROI mask (or landmark hull) out
        of the frame and resize to `self.roi_size`. Falls back to a
        centered crop if no face was found in this frame (keeps clip
        length consistent; such frames should be rare after good
        tracking and are implicitly down-weighted since the model sees
        a static/uninformative patch there)."""
        h, w = frame_bgr.shape[:2]
        if result.found and result.mask is not None:
            ys, xs = np.where(result.mask > 0)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            crop = frame_bgr[y0 : y1 + 1, x0 : x1 + 1]
        else:
            side = min(h, w) // 2
            cy, cx = h // 2, w // 2
            crop = frame_bgr[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]

        if crop.size == 0:
            crop = frame_bgr
        resized = cv2.resize(crop, self.roi_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _build_index(self) -> None:
        """Build (subject_idx, start_frame) pairs for all valid windows,
        without yet decoding video (frame *counts* only, via cv2 metadata)."""
        for subj_idx, subject in enumerate(self.subjects):
            if subject.video_path is not None:
                cap = cv2.VideoCapture(str(subject.video_path))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                n_frames = len(list(subject.frames_dir.glob("*.png")))

            for start in range(0, max(1, n_frames - self.clip_len_frames + 1), self.stride_frames):
                self._index.append((subj_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        subject_idx, start = self._index[i]
        roi_array, per_frame_hr = self._load_subject_frames_and_hr(subject_idx)

        end = min(start + self.clip_len_frames, roi_array.shape[0])
        clip = roi_array[start:end]
        hr_window = per_frame_hr[start:end]

        # Pad short trailing clips by repeating the last frame, rather
        # than dropping them (keeps every window usable, and the model
        # architecture expects a fixed T).
        if clip.shape[0] < self.clip_len_frames:
            pad_n = self.clip_len_frames - clip.shape[0]
            pad = np.repeat(clip[-1:], pad_n, axis=0)
            clip = np.concatenate([clip, pad], axis=0)
            hr_window = np.concatenate([hr_window, np.repeat(hr_window[-1:], pad_n)])

        clip_tensor = torch.from_numpy(clip).float() / 255.0  # (T, H, W, 3)
        clip_tensor = clip_tensor.permute(3, 0, 1, 2)  # -> (C, T, H, W) for Conv3d
        target_bpm = torch.tensor(float(np.mean(hr_window)), dtype=torch.float32)
        return clip_tensor, target_bpm
