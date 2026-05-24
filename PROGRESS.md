# FocusAI — Progress Summary

## Project Description
Create an intelligent focus and productivity monitoring system using webcam-based computer vision.

The system should detect:
- User presence
- Head/gaze direction
- Distractions
- Absence
- Possible phone usage
- Focus time

---

## Current Goal
Binary classifier that determines whether a person is **focused** or **distracted** from webcam frames, trained on custom-collected data — serving as the foundation for the broader monitoring system above.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep learning | TensorFlow 2.21 / Keras |
| Face detection & landmarks | MediaPipe Face Landmarker |
| Image processing | OpenCV |
| Classical ML / metrics | scikit-learn |
| Data collection server | Flask |
| Package manager | uv |

---

## Completed Steps

### 1. Data Collection
- **Tool:** Flask web app (`collection/app.py`) with browser UI (`collection/templates/index.html`).
- **How it works:** Captures webcam frames at **10 FPS** via `setInterval(captureFrame, 100)`, saves each frame as JPEG (quality 0.85) to `data/focused/` or `data/distracted/` depending on which button is held (or keyboard shortcut F/D).
- **Dataset size:** 2,336 images total — **997 focused**, **1,339 distracted**.
- **Image format:** JPEG, raw webcam resolution, mirrored horizontally before saving.

### 2. Preprocessing Pipeline (`src/preprocessing/crop_faces.py`)
- Downloads MediaPipe Face Landmarker model automatically if not present.
- For each raw image, detects face landmarks with MediaPipe.
- **Crops** the face region with 20% padding; falls back to full frame if no face is found.
- **Estimates head pose** (yaw, pitch, roll) via `solvePnP` using 6 canonical facial landmarks (nose tip, chin, eye corners, mouth corners).
- Saves cropped images to `data/cropped/focused/` and `data/cropped/distracted/` as JPEG (quality 95).
- Logs all results to `data/cropped/landmarks.csv` (columns: `filename`, `class`, `yaw`, `pitch`, `roll`, `face_detected`).
- Images where no face was detected are logged to `data/cropped/no_face_log.txt`; their angle columns are `null` in the CSV.
- Script is fully runnable via `python -m src.preprocessing.crop_faces` with a `main()` guard.

### 3. Legacy CNN Model (`notebooks/train_cnn.ipynb`) — do not modify
- Single-input sequential CNN trained on raw (non-cropped) images.
- 3× Conv2D + BatchNorm + MaxPool blocks → GlobalAveragePooling → Dense(128) → Dense(1, sigmoid).
- Data augmentation: random horizontal flip, ±5% rotation, ±5% zoom.
- Trained with Adam (lr=1e-3), early stopping (patience=7), ReduceLROnPlateau.
- **No class weights** — superseded by the dual-input model.

### 4. Dual-Input Model (`notebooks/train_dual.ipynb`) — current model
- Trains on **cropped face images + head-pose angles** simultaneously.
- Null angles in CSV (no face detected) are replaced with sentinel value `999.0` before training.
- Images and angles are joined by composite key `(class, filename)` to ensure alignment.
- **Architecture (`focus_dual`):**

```
Input 1 "image_input": (224, 224, 1) grayscale image
│
├─ Conv2D(32, 3×3, same, relu) → MaxPool(2×2)
├─ Conv2D(64, 3×3, same, relu) → MaxPool(2×2)
├─ Conv2D(128, 3×3, same, relu) → MaxPool(2×2)
├─ Flatten → Dense(128, relu) → Dropout(0.4)

Input 2 "angle_input": (3,) [yaw, pitch, roll]
│
├─ Dense(16, relu) → Dense(16, relu)

Concatenate → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)
Output: 0 = focused, 1 = distracted
```

- **Training settings:**

| Setting | Value |
|---|---|
| Optimizer | Adam (lr = 1e-3) |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience = 8, monitors val_loss, restores best weights |
| LR scheduler | ReduceLROnPlateau (factor = 0.5, patience = 4, min = 1e-6) |
| Class weights | Computed via `compute_class_weight("balanced")` to handle imbalance |
| Split | 80/20 stratified train/val |

- Model saved to `models/focus_model.h5` via `ModelCheckpoint`.

### 5. Project Infrastructure
- `src/utils/config.py` — single source of truth for all constants and paths (IMG_SIZE, CHANNELS, CLASSES, all data/model paths, MediaPipe model URL).
- `CLAUDE.md` — coding reference for the project (conventions, rules, architecture summary).
- `.gitignore` configured: `__pycache__`, `.pyc`, `models/`, notebook outputs excluded.

---

## Data

| Split | Focused | Distracted | Total |
|---|---|---|---|
| Raw images | 997 | 1,339 | 2,336 |
| After preprocessing | ~997 | ~1,339 | ~2,336 |

- Null angles (face not detected) → sentinel `999.0` at training time.
- Class imbalance (~43% focused / ~57% distracted) handled with `class_weight`.

---

## What's Next

1. **Run the dual-input notebook end-to-end** and record final validation metrics (accuracy, F1, confusion matrix).
2. **Real-time inference pipeline** — live webcam feed → MediaPipe landmarks → model prediction → overlay.
3. **Expand detection** — phone usage detection, absence detection, gaze direction.
