"""
Stage 1: Face detection & ROI tracking.

Uses MediaPipe's FaceLandmarker (the current Tasks API -- MediaPipe's
older `mp.solutions.face_mesh` API is deprecated and, as of mediapipe
>=0.10.30, may not even be present in the package) to locate a stable
forehead + both-cheeks region of interest (ROI) per frame, deliberately
avoiding eyes, mouth, and hair (all of which introduce motion artifacts
or occlusions that corrupt the rPPG signal). The ROI center is smoothed
with an exponential moving average (EMA) across frames to suppress
landmark-detection jitter without introducing much lag.

Setup note: FaceLandmarker requires a small (~3.7MB) model asset file
that is downloaded on first use to `models/face_landmarker.task` (see
`ensure_model_downloaded`). This requires one-time internet access to
storage.googleapis.com; if your environment blocks that domain, download
the file manually (URL printed in the error message) and place it at
`models/face_landmarker.task`.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mediapipe is required for face_tracking.py. Install with "
        "`pip install mediapipe`."
    ) from exc

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Landmark indices (MediaPipe Face Mesh, 468/478-point topology) that
# bound forehead and cheek patches. Chosen to stay clear of eyebrows,
# eyes, nose shadow, mouth, and hairline.
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109]
LEFT_CHEEK_IDX = [117, 118, 119, 120, 100, 142, 203, 206, 216]
RIGHT_CHEEK_IDX = [346, 347, 348, 349, 329, 371, 423, 426, 436]

ALL_ROI_IDX = FOREHEAD_IDX + LEFT_CHEEK_IDX + RIGHT_CHEEK_IDX


def build_roi_mask(landmarks_px: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Build the forehead+cheeks binary ROI mask from landmark pixel
    coordinates. Pulled out as a pure function (no MediaPipe dependency)
    so the ROI geometry can be unit-tested without a model file or a
    real face image.

    Args:
        landmarks_px: (len(ALL_ROI_IDX), 2) array of (x, y) pixel
            coordinates, in the same order as ALL_ROI_IDX.
        frame_shape: (height, width) of the target frame.

    Returns:
        (H, W) uint8 mask, 255 inside the ROI polygons, 0 elsewhere.
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for idx_group in (FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX):
        group_pts = landmarks_px[[ALL_ROI_IDX.index(i) for i in idx_group]]
        hull = cv2.convexHull(group_pts.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    return mask


def ensure_model_downloaded(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the FaceLandmarker model asset if not already present.

    Args:
        model_path: Where to store/find the .task model file.

    Returns:
        The path to the (now guaranteed-present) model file.

    Raises:
        RuntimeError: If the download fails (e.g. no internet access).
            The error message includes the URL for manual download.
    """
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download FaceLandmarker model from {MODEL_URL}.\n"
            f"If your environment has no internet access, download it manually "
            f"(e.g. on your laptop) and place it at: {model_path}\n"
            f"Original error: {exc}"
        ) from exc
    return model_path


@dataclass
class ROIResult:
    """Result of tracking one frame."""
    found: bool
    mean_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (R, G, B)
    mask: np.ndarray | None = None  # binary mask of the ROI, same HxW as frame
    landmarks_px: np.ndarray | None = None  # (N, 2) pixel coords of ROI landmarks


class FaceROITracker:
    """Tracks a forehead+cheeks ROI across a video stream.

    Usage:
        tracker = FaceROITracker()
        for frame in frames:  # frame is BGR, as read by cv2.VideoCapture
            result = tracker.process(frame, timestamp_ms=timestamp_ms)
            if result.found:
                r, g, b = result.mean_rgb
        tracker.close()
    """

    def __init__(self, smoothing_alpha: float = 0.4, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, model_path: Path | None = None):
        """
        Args:
            smoothing_alpha: EMA weight on the *new* landmark positions,
                in (0, 1]. Lower = smoother/more lag, higher = more
                responsive/more jitter. 0.4 is a reasonable default for
                30fps video.
            min_detection_confidence: MediaPipe face-detection threshold.
            min_tracking_confidence: MediaPipe landmark-tracking threshold.
            model_path: Path to the FaceLandmarker .task model file. If
                None, uses/downloads the default location.
        """
        resolved_path = ensure_model_downloaded(model_path or DEFAULT_MODEL_PATH)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.smoothing_alpha = smoothing_alpha
        self._smoothed_landmarks: np.ndarray | None = None  # (N, 2) EMA state
        self._frame_idx = 0
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> ROIResult:
        """Process a single BGR frame and return the ROI mean RGB.

        Args:
            frame_bgr: (H, W, 3) uint8 BGR frame, as produced by cv2.
            timestamp_ms: Monotonically increasing timestamp in
                milliseconds (required by MediaPipe's VIDEO running mode).
                If None, an internal frame counter is used assuming 30fps;
                pass real timestamps for non-uniform frame rates (e.g. a
                live webcam demo where frames may be dropped).

        Returns:
            ROIResult. If no face is found, `found=False` and callers
            should treat this frame as missing data (e.g. hold last
            value or mark the window as unreliable) rather than trusting
            mean_rgb, which will be all zeros.
        """
        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if timestamp_ms is None:
            timestamp_ms = int(self._frame_idx * (1000.0 / 30.0))
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        self._frame_idx += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return ROIResult(found=False)

        landmarks = result.face_landmarks[0]
        pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in ALL_ROI_IDX],
            dtype=np.float32,
        )

        # Exponential moving average smoothing to reduce per-frame jitter
        # from landmark-detection noise, without the lag of a long
        # sliding-window average.
        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = pts
        else:
            self._smoothed_landmarks = (
                self.smoothing_alpha * pts
                + (1 - self.smoothing_alpha) * self._smoothed_landmarks
            )
        smoothed_pts = self._smoothed_landmarks

        mask = build_roi_mask(smoothed_pts, (h, w))

        if mask.sum() == 0:
            return ROIResult(found=False)

        mean_rgb_bgr_order = cv2.mean(frame_bgr, mask=mask)[:3]  # (B, G, R)
        mean_rgb = np.array(
            [mean_rgb_bgr_order[2], mean_rgb_bgr_order[1], mean_rgb_bgr_order[0]]
        )  # -> (R, G, B)

        return ROIResult(found=True, mean_rgb=mean_rgb, mask=mask, landmarks_px=smoothed_pts)

    def reset(self) -> None:
        """Clear EMA smoothing and timestamp state (call between clips/videos)."""
        self._smoothed_landmarks = None
        self._frame_idx = 0
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceROITracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def extract_rgb_trace(video_path: str, tracker: FaceROITracker | None = None,
                       max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the tracker over an entire video file and return the RGB trace.

    Args:
        video_path: Path to a video file readable by OpenCV.
        tracker: An existing FaceROITracker, or None to create one.
        max_frames: Optional cap on number of frames processed.

    Returns:
        Tuple of:
            rgb_trace: (N, 3) array of mean R,G,B per frame.
            valid_mask: (N,) bool array, True where a face was found.
                Frames where no face was found have rgb_trace filled by
                forward-fill (or zeros if no prior valid frame exists) so
                downstream code always gets a fixed-length array; callers
                that care about reliability should check valid_mask.
            fps: Detected video frame rate.
    """
    own_tracker = tracker is None
    tracker = tracker or FaceROITracker()
    tracker.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rgb_values = []
    valid_flags = []
    last_valid = np.zeros(3)
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_count * (1000.0 / fps))
            result = tracker.process(frame, timestamp_ms=timestamp_ms)
            if result.found:
                last_valid = result.mean_rgb
                valid_flags.append(True)
            else:
                valid_flags.append(False)
            rgb_values.append(last_valid.copy())

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        if own_tracker:
            tracker.close()

    return np.array(rgb_values), np.array(valid_flags), fps
