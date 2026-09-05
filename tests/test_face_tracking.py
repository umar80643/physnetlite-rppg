"""Tests for src/face_tracking.py.

IMPORTANT SANDBOX LIMITATION: this development environment has no real
face photo/video and cannot reach storage.googleapis.com to download the
MediaPipe FaceLandmarker model file (blocked by network policy). That
means `FaceROITracker` itself -- and true face-detection accuracy --
could NOT be exercised end-to-end here.

What *is* tested here, without any model or network access:
  1. The ROI polygon/mask geometry (`build_roi_mask`), which is a pure
     function of landmark coordinates and is where a bug would silently
     corrupt the extracted RGB signal.
  2. That `ensure_model_downloaded` fails loudly and with an actionable
     message rather than silently, when the model can't be fetched.

Before trusting Stage 1 end-to-end, run it manually on your own machine
(with internet access, for the one-time model download) against a real
webcam frame or an UBFC-rPPG/PURE clip -- see the README section
"Validating Stage 1 yourself".
"""

import numpy as np
import pytest

from src.face_tracking import (
    ALL_ROI_IDX,
    FOREHEAD_IDX,
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
    build_roi_mask,
    ensure_model_downloaded,
)


def _fake_landmarks(frame_shape=(480, 640), face_center=(320, 240), face_radius=150):
    """Place ALL_ROI_IDX landmarks on a circle to approximate a face,
    purely so build_roi_mask has plausible, well-separated input points
    -- this does not need to look like a real face mesh topology."""
    n = len(ALL_ROI_IDX)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = face_center
    pts = np.stack(
        [cx + face_radius * np.cos(angles), cy + face_radius * np.sin(angles)], axis=1
    ).astype(np.float32)
    return pts


class TestBuildRoiMask:
    def test_mask_shape_matches_frame(self):
        frame_shape = (480, 640)
        pts = _fake_landmarks(frame_shape)
        mask = build_roi_mask(pts, frame_shape)
        assert mask.shape == frame_shape

    def test_mask_is_binary(self):
        frame_shape = (480, 640)
        pts = _fake_landmarks(frame_shape)
        mask = build_roi_mask(pts, frame_shape)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_mask_nonempty_for_valid_points(self):
        frame_shape = (480, 640)
        pts = _fake_landmarks(frame_shape)
        mask = build_roi_mask(pts, frame_shape)
        assert mask.sum() > 0

    def test_three_disjoint_regions_produce_three_separated_blobs(self):
        """Forehead and left/right cheeks should not accidentally merge
        into one connected blob if placed with realistic separation --
        catches an accidental index mix-up between the three groups."""
        frame_shape = (480, 640)
        # Forehead near top, cheeks left/right lower -- coarse standin
        # for real face geometry, well-separated so overlap would
        # indicate an indexing bug rather than expected adjacency.
        n_f, n_l, n_r = len(FOREHEAD_IDX), len(LEFT_CHEEK_IDX), len(RIGHT_CHEEK_IDX)
        pts = np.zeros((len(ALL_ROI_IDX), 2), dtype=np.float32)
        pts[: n_f] = np.array([320, 100]) + np.random.default_rng(0).normal(0, 5, (n_f, 2))
        pts[n_f : n_f + n_l] = np.array([150, 300]) + np.random.default_rng(1).normal(0, 5, (n_l, 2))
        pts[n_f + n_l :] = np.array([490, 300]) + np.random.default_rng(2).normal(0, 5, (n_r, 2))

        mask = build_roi_mask(pts, frame_shape)
        n_components, _ = _count_components(mask)
        assert n_components == 3

    def test_all_roi_idx_has_no_duplicate_indices_across_groups(self):
        """A landmark index appearing in two groups would silently
        double-count that pixel's contribution and is almost certainly
        a copy-paste bug."""
        groups = [FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX]
        seen = set()
        for group in groups:
            for idx in group:
                assert idx not in seen, f"Landmark {idx} appears in multiple ROI groups"
                seen.add(idx)
        assert seen == set(ALL_ROI_IDX)


def _count_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    import cv2

    n, labels = cv2.connectedComponents(mask)
    return n - 1, labels  # subtract background label


class TestModelDownload:
    def test_raises_actionable_error_when_unreachable(self, tmp_path, monkeypatch):
        """In this sandbox, storage.googleapis.com is not reachable, so
        this should fail loudly with a message telling the user exactly
        what to do -- never silently proceed with a missing model."""
        fake_path = tmp_path / "face_landmarker.task"
        try:
            ensure_model_downloaded(fake_path)
            pytest.skip("Model download unexpectedly succeeded in this environment.")
        except RuntimeError as e:
            assert "face_landmarker.task" in str(e)
            assert "manually" in str(e).lower()

    def test_skips_download_if_already_present(self, tmp_path):
        fake_path = tmp_path / "face_landmarker.task"
        fake_path.write_bytes(b"not a real model, just testing the cache check")
        result = ensure_model_downloaded(fake_path)
        assert result == fake_path
        assert result.read_bytes() == b"not a real model, just testing the cache check"
