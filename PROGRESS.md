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
Binary image classifier that determines whether a person is **focused** or **distracted** from webcam frames, trained on custom-collected data — serving as the foundation for the broader monitoring system above.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep learning | TensorFlow 2.21 / Keras |
| Classical ML / metrics | scikit-learn |
| Data collection server | Flask |
| Package manager | uv |

---

## Data Collection

- **Tool:** Flask web app (`collection/app.py`) with a browser UI (`collection/templates/index.html`).
- **How it works:** Captures webcam frames at **10 FPS** via `setInterval(captureFrame, 100)`, saves each frame as a JPEG (quality 0.85) to `data/focused/` or `data/distracted/` depending on which button is held (or keyboard shortcut F/D).
- **Dataset size:** 2,336 images total — **997 focused**, **1,339 distracted**.
- **Image format:** JPEG, raw webcam resolution captured in browser, mirrored horizontally before saving.

---

## Preprocessing

- Images resized to **224 × 224 pixels**.
- Converted to **grayscale** (1 channel).
- Pixel values normalized to `[0, 1]`.
- 80 / 20 stratified train-validation split → **1,868 train**, **468 val**.

---

## Model Architecture — `focus_cnn`

**Type:** Custom sequential CNN for binary classification.  
**Total parameters:** 110,209 (~430 KB)

```
Input: (224, 224, 1)
│
├─ Data augmentation (training only)
│   ├─ RandomFlip (horizontal)
│   ├─ RandomRotation (±5%)
│   └─ RandomZoom (±5%)
│
├─ Block 1: Conv2D(32, 3×3, same) → BatchNorm → MaxPool(2×2)  → (112, 112, 32)
├─ Block 2: Conv2D(64, 3×3, same) → BatchNorm → MaxPool(2×2)  → (56, 56, 64)
├─ Block 3: Conv2D(128, 3×3, same) → BatchNorm → MaxPool(2×2) → (28, 28, 128)
│
├─ GlobalAveragePooling2D → (128,)
├─ Dropout(0.4)
├─ Dense(128, relu)
├─ Dropout(0.3)
└─ Dense(1, sigmoid)  ← 0 = focused, 1 = distracted
```

---

## Training

| Setting | Value |
|---|---|
| Optimizer | Adam (lr = 1e-3) |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience = 7 (monitors val_loss, restores best weights) |
| LR scheduler | ReduceLROnPlateau (factor = 0.5, patience = 3, min = 1e-6) |

---

## Output

- Trained model saved to `models/focus_model.h5`.
- Notebook includes evaluation cell with classification report, confusion matrix, and a single-image inference demo.

---

## What's Next

1. **Re-train on `data/cropped/`** — use the face-cropped images instead of raw frames to reduce background noise.
2. **Fix class imbalance** — add `class_weight` to `model.fit()` (997 focused vs 1,339 distracted).
3. **Real-time inference pipeline** — live webcam feed → prediction overlay.
