"""
Real-time focus monitor Flask app (port 5001).

Receives webcam frames as base64 JPEG, replicates the exact training
preprocessing pipeline (MediaPipe face detection, crop, head pose
estimation via src/preprocessing/crop_faces.py) and feeds the dual-input
model with raw inputs: grayscale image (224, 224, 1) in [0, 255] and
head pose angles (3,) with 999.0 as the no-face sentinel.

Predictions are temporally smoothed over a sliding window (~2.5 s at
10 FPS) before being returned to the dashboard.
"""

import base64
import sys
import threading
import webbrowser
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.crop_faces import (
    build_detector,
    crop_face,
    download_model,
    estimate_head_pose,
)
from src.utils.config import (
    CHANNELS,
    IMG_SIZE,
    MEDIAPIPE_MODEL_PATH,
    MEDIAPIPE_MODEL_URL,
    MODEL_PATH,
)

NO_FACE_ANGLE = 999.0          # sentinel used at training time for null angles
SMOOTHING_WINDOW = 25          # ~2.5 s of predictions at 10 FPS

app = Flask(__name__)

_model_path = PROJECT_ROOT / MODEL_PATH

try:
    import tensorflow as tf

    if _model_path.exists():
        model = tf.keras.models.load_model(str(_model_path))
    else:
        model = None
except Exception:
    model = None

# Same MediaPipe detector used by the training preprocessing pipeline.
try:
    download_model(PROJECT_ROOT / MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_URL)
    detector = build_detector(PROJECT_ROOT / MEDIAPIPE_MODEL_PATH)
except Exception:
    detector = None

# Sliding window of raw scores for temporal smoothing.
# detector/model are not thread-safe, so inference is serialized.
_scores: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
_lock = threading.Lock()


def preprocess_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Replicate the training preprocessing (crop_faces) on a single frame.

    Detects the face with MediaPipe, crops it and estimates head pose.
    Falls back to the full frame with 999.0 angles when no face is found,
    exactly like the training pipeline does.

    Args:
        frame: BGR image as a numpy array.

    Returns:
        Tuple of:
        - image array (1, 224, 224, 1) float32, grayscale, raw [0, 255]
        - angles array (1, 3) float32 as [yaw, pitch, roll]
        - face_detected flag
    """
    face_detected = False
    final = frame
    yaw, pitch, roll = NO_FACE_ANGLE, NO_FACE_ANGLE, NO_FACE_ANGLE

    if detector is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            cropped = crop_face(frame, landmarks)
            final = cropped if cropped is not None else frame
            yaw, pitch, roll = estimate_head_pose(landmarks, frame.shape)
            face_detected = True

    resized = cv2.resize(final, IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image_array = gray.astype(np.float32).reshape(1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
    angles = np.array([[yaw, pitch, roll]], dtype=np.float32)
    return image_array, angles, face_detected


@app.route("/")
def index():
    """Serve the monitor dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Run dual-input focus inference on a base64-encoded JPEG frame.

    Expects JSON: {"image": "data:image/jpeg;base64,..."}
    Returns JSON: {"label", "score", "raw_score", "face_detected"} where
    label/score are temporally smoothed over the last SMOOTHING_WINDOW
    predictions.
    """
    if model is None:
        return jsonify(
            {"label": "no_model", "score": 0.5, "raw_score": 0.5, "face_detected": False}
        )

    data = request.json
    _, encoded = data["image"].split(",", 1)
    img_bytes = base64.b64decode(encoded)

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    with _lock:
        image_array, angles, face_detected = preprocess_frame(frame)
        raw_score = float(
            model.predict(
                {"image_input": image_array, "angle_input": angles}, verbose=0
            )[0][0]
        )
        _scores.append(raw_score)
        smoothed = float(np.mean(_scores))

    label = "distracted" if smoothed > 0.5 else "focused"

    return jsonify(
        {
            "label": label,
            "score": round(smoothed, 3),
            "raw_score": round(raw_score, 3),
            "face_detected": face_detected,
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Clear the temporal smoothing window (new monitoring session)."""
    with _lock:
        _scores.clear()
    return jsonify({"ok": True})


@app.route("/health")
def health():
    """Return model and detector load status."""
    return jsonify(
        {
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "detector_loaded": detector is not None,
        }
    )


def run(port: int = 5001) -> None:
    """Open the browser and start the Flask server on the given port."""
    webbrowser.open(f"http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    run()
