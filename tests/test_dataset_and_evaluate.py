"""Integration-style tests for src/dataset.py (discovery + splitting) and
src/evaluate.py (ground-truth BPM windowing), using synthetic UBFC-rPPG-
style folders so they run without the real dataset.

Model-dependent pieces (RPPGWindowDataset, which needs FaceROITracker /
the FaceLandmarker model file) are NOT exercised here for the same
sandbox reason noted in test_face_tracking.py -- no network access to
the model file in this environment. Structural discovery/splitting logic,
which is pure Python + OpenCV metadata reads, IS fully tested.
"""

import numpy as np
import pytest

from src.dataset import SubjectRecord, discover_subjects, subject_level_split
from src.evaluate import compute_gt_bpm_windows


def _write_synthetic_ubfc_subject(root, subject_name: str, n_frames: int = 300, fps: float = 30.0,
                                   bpm: float = 72.0, seed: int = 0):
    """Write a minimal synthetic UBFC-rPPG-style subject folder: a tiny
    solid-color video (face detection will fail on it, which is fine --
    these tests target discovery/parsing, not detection) plus a
    ground_truth.txt with a synthetic PPG waveform and matching per-frame
    HR."""
    import cv2

    subject_dir = root / "UBFC-rPPG" / subject_name
    subject_dir.mkdir(parents=True)

    video_path = subject_dir / "vid.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (64, 64))
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        frame = (rng.integers(0, 255, (64, 64, 3))).astype(np.uint8)
        writer.write(frame)
    writer.release()

    t = np.arange(n_frames) / fps
    waveform = np.sin(2 * np.pi * (bpm / 60.0) * t)
    hr = np.full(n_frames, bpm)
    gt_path = subject_dir / "ground_truth.txt"
    gt_path.write_text(
        " ".join(f"{v:.4f}" for v in waveform) + "\n"
        + " ".join(f"{v:.2f}" for v in hr) + "\n"
        + " ".join(f"{v:.4f}" for v in t) + "\n"
    )
    return subject_dir


class TestDiscoverSubjects:
    def test_discovers_synthetic_ubfc_subjects(self, tmp_path):
        for i, name in enumerate(["subject1", "subject3", "subject5"]):
            _write_synthetic_ubfc_subject(tmp_path, name, seed=i)

        subjects = discover_subjects(tmp_path)
        assert len(subjects) == 3
        assert {s.subject_id for s in subjects} == {"subject1", "subject3", "subject5"}
        assert all(s.dataset == "UBFC-rPPG" for s in subjects)

    def test_skips_incomplete_subject_folders(self, tmp_path):
        _write_synthetic_ubfc_subject(tmp_path, "good_subject")
        incomplete_dir = tmp_path / "UBFC-rPPG" / "incomplete_subject"
        incomplete_dir.mkdir(parents=True)  # no video, no ground_truth.txt

        subjects = discover_subjects(tmp_path)
        assert len(subjects) == 1
        assert subjects[0].subject_id == "good_subject"

    def test_returns_empty_list_when_no_datasets_present(self, tmp_path):
        subjects = discover_subjects(tmp_path)
        assert subjects == []

    def test_parses_ground_truth_correctly(self, tmp_path):
        _write_synthetic_ubfc_subject(tmp_path, "subject1", n_frames=90, bpm=75.0)
        subjects = discover_subjects(tmp_path)
        s = subjects[0]
        assert len(s.gt_waveform) == 90
        assert np.allclose(s.gt_hr_bpm, 75.0)
        assert s.gt_timestamps_sec[-1] == pytest.approx(89 / 30.0, abs=1e-3)


class TestSubjectLevelSplit:
    def _make_fake_subjects(self, n: int) -> list[SubjectRecord]:
        return [
            SubjectRecord(
                subject_id=f"s{i}", dataset="UBFC-rPPG", video_path=None, frames_dir=None,
                fps=30.0, gt_waveform=np.zeros(10), gt_hr_bpm=np.full(10, 70.0),
                gt_timestamps_sec=np.arange(10) / 30.0,
            )
            for i in range(n)
        ]

    def test_split_covers_all_subjects_no_overlap(self):
        subjects = self._make_fake_subjects(10)
        splits = subject_level_split(subjects, seed=1)
        all_ids = set()
        for part in splits.values():
            ids = {s.subject_id for s in part}
            assert not (ids & all_ids), "Overlap between splits -- subject leakage!"
            all_ids |= ids
        assert all_ids == {s.subject_id for s in subjects}

    def test_split_is_reproducible_with_same_seed(self):
        subjects = self._make_fake_subjects(10)
        splits_a = subject_level_split(subjects, seed=7)
        splits_b = subject_level_split(subjects, seed=7)
        for key in ("train", "val", "test"):
            ids_a = [s.subject_id for s in splits_a[key]]
            ids_b = [s.subject_id for s in splits_b[key]]
            assert ids_a == ids_b

    def test_raises_on_empty_subject_list(self):
        with pytest.raises(ValueError):
            subject_level_split([])


class TestComputeGtBpmWindows:
    def test_uses_direct_hr_labels_when_available(self):
        n = 300
        fps = 30.0
        t = np.arange(n) / fps
        subject = SubjectRecord(
            subject_id="s1", dataset="UBFC-rPPG", video_path=None, frames_dir=None, fps=fps,
            gt_waveform=np.zeros(n), gt_hr_bpm=np.linspace(60, 90, n), gt_timestamps_sec=t,
        )
        win_starts, gt_bpm = compute_gt_bpm_windows(subject, window_sec=5.0, stride_sec=1.0)
        assert len(win_starts) == len(gt_bpm)
        assert len(win_starts) > 0
        # HR increases monotonically in this synthetic subject.
        assert np.all(np.diff(gt_bpm) >= -1e-6)

    def test_derives_bpm_from_waveform_when_no_hr_labels(self):
        fps = 30.0
        duration = 20
        n = int(duration * fps)
        t = np.arange(n) / fps
        bpm_true = 80.0
        waveform = np.sin(2 * np.pi * (bpm_true / 60.0) * t)
        subject = SubjectRecord(
            subject_id="s2", dataset="UBFC-rPPG", video_path=None, frames_dir=None, fps=fps,
            gt_waveform=waveform, gt_hr_bpm=None, gt_timestamps_sec=None,
        )
        win_starts, gt_bpm = compute_gt_bpm_windows(subject, window_sec=8.0, stride_sec=2.0)
        assert len(gt_bpm) > 0
        assert np.all(np.abs(gt_bpm - bpm_true) < 3.0)
