import base64
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import CHANNELS, IMG_SIZE, MODEL_PATH

app = Flask(__name__)

_model_path = Path(__file__).resolve().parent.parent / MODEL_PATH

try:
    import tensorflow as tf

    if _model_path.exists():
        model = tf.keras.models.load_model(str(_model_path))
    else:
        model = None
except Exception:
    model = None


@app.route("/")
def index():
    """Serve the monitor dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Run focus inference on a base64-encoded JPEG frame.

    Expects JSON: {"image": "data:image/jpeg;base64,..."}
    Returns JSON: {"label": "focused"|"distracted"|"no_model", "score": float}
    """
    if model is None:
        return jsonify({"label": "no_model", "score": 0.5})

    data = request.json
    _, encoded = data["image"].split(",", 1)
    img_bytes = base64.b64decode(encoded)

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    image_array = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], CHANNELS)

    score = float(model.predict(image_array, verbose=0)[0][0])
    label = "distracted" if score > 0.5 else "focused"

    return jsonify({"label": label, "score": round(score, 3)})


@app.route("/health")
def health():
    """Return model load status."""
    return jsonify({"model_loaded": model is not None, "model_path": MODEL_PATH})


def run(port: int = 5001) -> None:
    """Open the browser and start the Flask server on the given port."""
    webbrowser.open(f"http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    run()
