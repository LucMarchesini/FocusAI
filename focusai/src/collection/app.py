import base64
import datetime
import sys
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DATA_RAW_PATH

app = Flask(__name__)
BASE_PATH = Path(DATA_RAW_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/save_frame", methods=["POST"])
def save_frame():
    data = request.json
    label = data["label"]  # "focused" or "distracted"
    image_data = data["image"]  # base64-encoded JPEG data URL

    if label not in ("focused", "distracted"):
        return jsonify({"status": "error", "message": "Invalid label"}), 400

    _, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    folder = BASE_PATH / label
    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (folder / f"{timestamp}.jpg").write_bytes(img_bytes)

    counts = _get_counts()
    return jsonify({"status": "ok", "counts": counts})


@app.route("/counts")
def counts():
    return jsonify(_get_counts())


def _get_counts():
    return {
        "focused": len(list((BASE_PATH / "focused").glob("*.jpg"))),
        "distracted": len(list((BASE_PATH / "distracted").glob("*.jpg"))),
    }


def run(port=5000):
    webbrowser.open(f"http://localhost:{port}")
    app.run(debug=False, port=port)


if __name__ == "__main__":
    run()
