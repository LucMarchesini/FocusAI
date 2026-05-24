"""
Face cropping and head pose estimation pipeline.
Processes raw training images using MediaPipe Face Landmarker,
crops the face region, computes head pose angles, and saves results
to data/cropped/ preserving the focused/distracted class structure.
"""

from __future__ import annotations
import csv
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def download_model(model_path: Path, url: str) -> None:
    """
    Download the MediaPipe Face Landmarker model if not already present.

    Args:
        model_path: Destination path where the model file will be saved.
        url: Download URL for the model.
    """
    if model_path.exists():
        print("Model already exists, skipping download.")
        return
    print("Downloading MediaPipe Face Landmarker model...")
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")


def crop_face(
    image: np.ndarray,
    landmarks,
    padding: float = 0.2,
) -> np.ndarray | None:
    """
    Crop the face region from an image using MediaPipe landmark coordinates.

    Args:
        image: BGR image as a numpy array.
        landmarks: Face landmarks result from FaceLandmarker (face_landmarks[0]).
        padding: Fractional padding applied to each side of the bounding box.

    Returns:
        Cropped BGR image, or None if landmarks is empty or None.
    """
    if not landmarks:
        return None

    h, w = image.shape[:2]
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    return image[y1:y2, x1:x2]


def estimate_head_pose(
    landmarks,
    image_shape: tuple,
) -> tuple[float, float, float]:
    """
    Estimate head pose (yaw, pitch, roll) from 6 facial landmarks using solvePnP.

    Args:
        landmarks: Face landmarks result from FaceLandmarker (face_landmarks[0]).
        image_shape: Shape of the original image as (height, width, channels).

    Returns:
        Tuple of (yaw, pitch, roll) in degrees, rounded to 2 decimal places.
        Returns (0.0, 0.0, 0.0) if estimation fails.
    """
    try:
        # Step 1 — define the 6 landmark indices to use
        LANDMARK_INDICES = {
            "nose_tip":           1,
            "chin":               152,
            "left_eye_corner":    263,
            "right_eye_corner":   33,
            "left_mouth_corner":  287,
            "right_mouth_corner": 57,
        }

        # Step 2 — canonical 3D face model reference points (in mm)
        model_points = np.array([
            (0.0,    0.0,    0.0),    # nose tip
            (0.0,  -63.6,  -12.5),   # chin
            (-43.3,  32.7,  -26.0),  # left eye outer corner
            ( 43.3,  32.7,  -26.0),  # right eye outer corner
            (-28.9, -28.9,  -24.1),  # left mouth corner
            ( 28.9, -28.9,  -24.1),  # right mouth corner
        ], dtype=np.float64)

        # Step 3 — extract 2D image points from landmarks
        h, w = image_shape[:2]
        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in LANDMARK_INDICES.values()
        ], dtype=np.float64)

        # Step 4 — estimate camera intrinsics from image size
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # Step 5 — solve for rotation vector using solvePnP
        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return (0.0, 0.0, 0.0)

        # Step 6 — convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Step 7 — extract Euler angles from rotation matrix
        pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw   = np.degrees(np.arctan2(-rotation_matrix[2, 0],
                    np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
        roll  = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

        return (round(yaw, 2), round(pitch, 2), round(roll, 2))

    except Exception:
        return (0.0, 0.0, 0.0)


def append_to_csv(csv_path: Path, rows: list[dict]) -> None:
    """
    Append rows to the landmarks CSV file. Creates the file with a header if it doesn't exist.

    Args:
        csv_path: Path to the CSV file.
        rows: List of dicts with keys: filename, class, yaw, pitch, roll, face_detected.
    """
    columns = ["filename", "class", "yaw", "pitch", "roll", "face_detected"]
    file_exists = csv_path.exists()
    mode = "a" if file_exists else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def log_no_face(log_path: Path, filename: str) -> None:
    """
    Append a filename to the no-face log file.

    Args:
        log_path: Path to the log file.
        filename: The image filename to log (one per line).
    """
    with open(log_path, "a") as f:
        f.write(filename + "\n")


def build_detector(model_path: Path) -> vision.FaceLandmarker:
    """
    Instantiate and return a MediaPipe FaceLandmarker.

    Args:
        model_path: Path to the .task model file.

    Returns:
        A configured FaceLandmarker instance with num_faces=1.
    """
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_category(
    category: str,
    input_dir: Path,
    output_dir: Path,
    detector: vision.FaceLandmarker,
    csv_path: Path,
    log_path: Path,
    output_size: tuple = (224, 224),
) -> dict:
    """
    Process all images in one class folder: crop faces, estimate head pose,
    save cropped images, and log results to CSV.

    Args:
        category: "focused" or "distracted".
        input_dir: Folder containing raw .jpg images for this class.
        output_dir: Folder where cropped images will be saved (data/cropped/<category>/).
        detector: Initialized FaceLandmarker instance.
        csv_path: Path to landmarks.csv.
        log_path: Path to no_face_log.txt.
        output_size: Target (width, height) for saved images.

    Returns:
        Dict with keys: total, cropped, fallbacks.
    """
    images = list(input_dir.glob("*.jpg"))
    total = len(images)
    cropped_count = 0
    fallbacks = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images, start=1):
        filename = img_path.name
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error reading {img_path}, skipping.")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            cropped = crop_face(image, landmarks)
            yaw, pitch, roll = estimate_head_pose(landmarks, image.shape)
            face_detected = True
            cropped_count += 1
            final = cropped if cropped is not None else image
        else:
            final = image
            yaw, pitch, roll = None, None, None
            face_detected = False
            fallbacks += 1
            log_no_face(log_path, filename)

        final = cv2.resize(final, output_size)
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])

        append_to_csv(csv_path, [{
            "filename": f"data/cropped/{category}/{filename}",
            "class": category,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "face_detected": face_detected,
        }])

        if face_detected:
            print(f"[{i}/{total}] {category} — {filename} — yaw: {yaw}°, pitch: {pitch}°, roll: {roll}°")
        else:
            print(f"[{i}/{total}] {category} — {filename} — sem face detectada")

    return {"total": total, "cropped": cropped_count, "fallbacks": fallbacks}


def main() -> None:
    """
    Full pipeline: download model, build detector, process both classes, print summary.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.config import (
        MEDIAPIPE_MODEL_PATH,
        MEDIAPIPE_MODEL_URL,
        DATA_RAW_PATH,
        DATA_CROPPED_PATH,
        LANDMARK_CSV_PATH,
        NO_FACE_LOG_PATH,
    )

    download_model(PROJECT_ROOT / MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_URL)
    detector = build_detector(PROJECT_ROOT / MEDIAPIPE_MODEL_PATH)

    (PROJECT_ROOT / DATA_CROPPED_PATH / "focused").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / DATA_CROPPED_PATH / "distracted").mkdir(parents=True, exist_ok=True)

    results = {}
    for category in ("focused", "distracted"):
        results[category] = process_category(
            category=category,
            input_dir=PROJECT_ROOT / DATA_RAW_PATH / category,
            output_dir=PROJECT_ROOT / DATA_CROPPED_PATH / category,
            detector=detector,
            csv_path=PROJECT_ROOT / LANDMARK_CSV_PATH,
            log_path=PROJECT_ROOT / NO_FACE_LOG_PATH,
        )

    f = results["focused"]
    d = results["distracted"]
    print(
        f"\n╔══════════════╦═══════╦═════════╦═══════════╗\n"
        f"║ Class        ║ Total ║ Cropped ║ Fallbacks ║\n"
        f"╠══════════════╬═══════╬═════════╬═══════════╣\n"
        f"║ focused      ║ {f['total']:>5} ║ {f['cropped']:>7} ║ {f['fallbacks']:>9} ║\n"
        f"║ distracted   ║ {d['total']:>5} ║ {d['cropped']:>7} ║ {d['fallbacks']:>9} ║\n"
        f"╚══════════════╩═══════╩═════════╩═══════════╝"
    )


if __name__ == "__main__":
    main()
