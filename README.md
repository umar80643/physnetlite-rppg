# Remote Photoplethysmography (rPPG) Heart Rate Estimation from Video

Estimate a person's heart rate (BPM) purely from RGB face video — no contact
sensor — by detecting the subtle skin-color fluctuations caused by blood 
flow. This project implements:

1. **Face detection & ROI tracking** (MediaPipe FaceLandmarker)
2. **A classical signal-processing baseline** (CHROM and POS)
3. **A deep-learning model** (a compact PhysNet-style 3D-CNN, "PhysNetLite")
4. **Quantitative evaluation** against pulse-oximeter ground truth (MAE, RMSE,
   Pearson r, Bland-Altman)
5. **A real-time webcam demo**

---

## ⚠️ Read this first: what has and hasn't been verified

This repo was built and tested in a sandboxed environment with **no real
face video, no webcam, and no access to UBFC-rPPG/PURE or Google's model
CDN**. To be transparent about exactly what that means:

| Component | Status |
|---|---|
| Signal processing (`utils.py`): detrend, bandpass filter, FFT-BPM extraction, metrics | ✅ 25 unit tests pass, incl. recovering known BPMs (45–180) from synthetic signals with noise |
| Classical baseline (`classical_methods.py`): CHROM, POS | ✅ 10 unit tests pass; recovers synthetic BPMs 50–150 with **MAE < 0.2 BPM** (see `results/plots/synthetic_classical_baseline.png`) |
| Face ROI geometry (`face_tracking.py`) | ✅ Mask/polygon logic unit-tested (shape, no index overlap, 3 disjoint regions). ⚠️ **Not tested against a real face** — no face image was available in the sandbox |
| Face detection model download | ⚠️ **Fails in this sandbox** (network policy blocks `storage.googleapis.com`). Fails loudly with instructions, doesn't silently proceed. **You must run this on a machine with normal internet access at least once.** |
| Dataset loader (`dataset.py`): subject discovery, ground-truth parsing, subject-level split | ✅ 9 tests pass against synthetic UBFC-rPPG-formatted folders |
| Deep model (`model.py`): PhysNetLite + negative-Pearson loss | ✅ 8 tests pass: correct shapes, gradients reach every parameter, loss is scale/shift invariant, <2M params |
| Training loop end-to-end | ✅ Verified on synthetic clips — loss drops from 1.15 → 0.0002 over 60 steps and the model's FFT-recovered BPM converges to within ~2–4 BPM of the true synthetic value (`results/plots/synthetic_training_loss.png`) |
| Real dataset training/eval numbers | ❌ **Not run** — UBFC-rPPG/PURE require an academic-use request form (see below); no such data exists in this sandbox |
| Live webcam demo | ❌ **Not run** — no webcam / display in this sandbox. Code is written and follows the same API as the (tested) classical/model code paths, but you should test it yourself before relying on it |

**Bottom line:** every stage that could be verified without real face video
or a live camera has been — with actual passing unit tests and an actual
synthetic end-to-end run, not just code that "looks right." The two things
that fundamentally require your machine (a real face and a webcam) are
implemented but unverified, and are flagged everywhere in the code and
below so you know exactly what to check first.

Run `python notebooks/synthetic_pipeline_smoke_test.py` yourself to
reproduce the synthetic validation above.

---

## Project structure

```
rppg-project/
├── data/
│   ├── raw/                      # you populate this — see "Dataset setup"
│   └── processed/
├── src/
│   ├── face_tracking.py          # Stage 1: MediaPipe ROI tracking
│   ├── classical_methods.py      # Stage 2: CHROM, POS
│   ├── model.py                  # Stage 3: PhysNetLite (3D-CNN) + loss
│   ├── dataset.py                # UBFC-rPPG/PURE loader, subject-level split
│   ├── train.py                  # Stage 3: training script
│   ├── evaluate.py               # Stage 4: metrics + plots
│   └── utils.py                  # filtering, FFT-BPM, metrics (core primitives)
├── demo/
│   └── live_demo.py              # Stage 5: real-time webcam demo
├── notebooks/
│   └── synthetic_pipeline_smoke_test.py   # no-dataset, no-webcam sanity check
├── tests/                        # 59 unit/integration tests, see table above
├── results/
│   ├── metrics.json              # written by evaluate.py
│   └── plots/
├── models/                       # FaceLandmarker .task file goes here
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note on OpenCV:** `requirements.txt` doesn't pin `opencv-python` vs.
`opencv-python-headless`. Install plain `opencv-python` (not `-headless`) if
you want the live webcam demo's `cv2.imshow` window — headless builds have
no GUI support.

### One-time: download the face-tracking model

`face_tracking.py` auto-downloads MediaPipe's `face_landmarker.task`
(~3.7MB) from `storage.googleapis.com` the first time you run anything that
needs it. This just needs to work once, from a machine with normal internet
access:

```bash
python -c "from src.face_tracking import ensure_model_downloaded; ensure_model_downloaded()"
```

If that domain is blocked on your network, download it manually from the
URL printed in the error message and place it at
`models/face_landmarker.task`.

### Validating Stage 1 yourself

Since this couldn't be tested against a real face in the build environment,
do this before trusting it:

```python
import cv2
from src.face_tracking import FaceROITracker

cap = cv2.VideoCapture(0)  # or a video file path
tracker = FaceROITracker()
ok, frame = cap.read()
result = tracker.process(frame, timestamp_ms=0)
print("Face found:", result.found)

if result.found:
    cv2.imshow("ROI", cv2.bitwise_and(frame, frame, mask=result.mask))
    cv2.waitKey(0)
```

You should see a mask covering the forehead and both cheeks, avoiding eyes,
eyebrows, mouth, and hairline. If the ROI looks off, the landmark index
groups (`FOREHEAD_IDX`, `LEFT_CHEEK_IDX`, `RIGHT_CHEEK_IDX`) in
`face_tracking.py` are the place to adjust — they're MediaPipe's standard
468-point face-mesh topology indices.

---

## Dataset setup

This project uses **UBFC-rPPG** and/or **PURE**, both of which require
filling out a short academic-use request form with the original authors —
they cannot be auto-downloaded. Budget a day or two for approval.

- **UBFC-rPPG**: request access via the dataset authors' page (search
  "UBFC-rPPG dataset" — hosted by the Université de Bourgogne Franche-Comté
  VANTH/LE2I group). Place subjects at:
  ```
  data/raw/UBFC-rPPG/subject1/vid.avi
  data/raw/UBFC-rPPG/subject1/ground_truth.txt
  data/raw/UBFC-rPPG/subject3/...
  ```
  `ground_truth.txt` format (space-separated, 3 lines): PPG waveform, per-frame HR (BPM), timestamps (s).

- **PURE**: request access via the dataset authors' page (TU Ilmenau). Place subjects at:
  ```
  data/raw/PURE/01-01/01-01/*.png        # frame sequence
  data/raw/PURE/01-01/01-01.json         # pulse waveform + per-frame HR
  ```

`src/dataset.py`'s `discover_subjects()` auto-detects whichever of the two
you've populated (or both) and silently skips the other, so you can develop
against just one.

---

## Running each stage

### Stage 1+2: face tracking + classical baseline (no training needed)

```bash
python -c "
from src.face_tracking import extract_rgb_trace
from src.classical_methods import chrom
from src.utils import estimate_hr_fft

rgb_trace, valid_mask, fps = extract_rgb_trace('path/to/video.avi')
print(f'Face found in {valid_mask.mean()*100:.0f}% of frames')
pulse = chrom(rgb_trace, fps=fps)
est = estimate_hr_fft(pulse, fps=fps)
print(f'Estimated HR: {est.bpm:.1f} BPM (confidence {est.confidence:.2f})')
"
```

### Stage 3: train the deep-learning model

```bash
python -m src.train --data-root data/raw --epochs 30 --batch-size 4
```

Key flags: `--clip-len` (frames per training clip, default 150 = 5s @
30fps), `--stride` (window hop), `--roi-size`, `--device` (auto-detects
CUDA). Saves the best checkpoint (by validation MAE) to
`results/checkpoints/best_model.pt` and a full loss/MAE history to
`results/train_history.json`.

**Note on the training target:** UBFC-rPPG/PURE provide full ground-truth
PPG *waveforms*, but `dataset.py` simplifies this to a per-frame *HR
scalar* to keep the pipeline smaller. `train.py` then synthesizes a target
sinusoid at that HR to train against with the negative-Pearson loss. This
is a reasonable, standard simplification, but if you want a stronger
baseline, the natural extension is to resample the real ground-truth
waveform onto each clip's frame timestamps and train against that directly
— `dataset.py`'s `SubjectRecord.gt_waveform` already holds the raw waveform
for this purpose.

### Stage 4: evaluate

```bash
python -m src.evaluate --data-root data/raw --checkpoint results/checkpoints/best_model.pt
```

Produces `results/metrics.json` (MAE / RMSE / Pearson r per method) and, in
`results/plots/`: predicted-vs-ground-truth BPM traces per test subject and
a Bland-Altman plot per method. By default this runs the classical
CHROM/POS baseline (`--methods CHROM POS`); see the "Extending evaluate.py"
note printed by the script for wiring in the trained model's predictions
the same way.

### Stage 5: live webcam demo

```bash
python -m demo.live_demo                                  # CHROM baseline
python -m demo.live_demo --method POS
python -m demo.live_demo --model-checkpoint results/checkpoints/best_model.pt
```

Shows a live BPM overlay, updated once per second from a rolling 10-second
buffer. Displays "No face detected" or "Signal unstable" (low FFT peak
confidence) instead of a garbage number when tracking is poor. Press `q` to
quit. **Not runnable/testable in the build sandbox** — try it on your own
machine before relying on it, and tune `--min-confidence` if it's too
trigger-happy or not cautious enough for your lighting.

### Reproducing the synthetic validation (no dataset/webcam needed)

```bash
python notebooks/synthetic_pipeline_smoke_test.py
```

### Running the test suite

```bash
python -m pytest tests/ -v
```

---

## Design notes / assumptions

- **Frame rate**: both UBFC-rPPG and PURE are recorded near 30 fps; the
  code reads actual fps from video metadata where available (UBFC-rPPG) and
  assumes 30 fps for PURE's frame sequences (per the dataset's protocol).
- **HR search band**: 0.7–4.0 Hz (42–240 BPM), the standard rPPG range
  covering resting through high-intensity-exercise heart rates.
- **Subject-level splitting**: train/val/test splits are by *subject*, not
  frame — critical to avoid identity leakage (see `dataset.py` docstring).
- **ROI choice**: forehead + both cheeks, avoiding eyes (blinking),
  eyebrows/mouth (motion), and hair (no perfusion) — the standard ROI in
  the rPPG literature.
- **Model size**: PhysNetLite is <2M parameters (unit-tested), trainable on
  CPU or a single consumer GPU, per the project's goal of a correct,
  well-evaluated pipeline over architectural complexity.

## References

- de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based
  rPPG. *IEEE Trans. Biomedical Engineering.*
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
  Algorithmic principles of remote-PPG. *IEEE Trans. Biomedical
  Engineering.*
- Yu, Z., Li, X., & Zhao, G. (2019). Remote Photoplethysmograph Signal
  Measurement from Facial Videos Using Spatio-Temporal Networks. *BMVC.*
- Tarvainen, M. P., Ranta-aho, P. O., & Karjalainen, P. A. (2002). An
  advanced detrending method with application to HRV analysis. *IEEE Trans.
  Biomedical Engineering.*
