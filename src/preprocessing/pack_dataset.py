"""
Dataset packing pipeline.

Pre-packs the cropped face dataset into a single compressed ``.npz`` file so
that training on Google Colab no longer needs the Google Drive mount nor the
per-image JPEG decoding. The loading and alignment logic here mirrors EXACTLY
the notebook cells "## 1 Carregamento das Imagens", "## 2 Carregamento dos
Ângulos" and the array-building part of "## 3 Pré-processamento" in
``notebooks/train_dual.ipynb``.

Images are stored as raw ``uint8`` in ``[0, 255]`` on purpose: ``uint8`` is the
native pixel range, so this is lossless and keeps the ``.npz`` ~4x smaller than
float32. The cast to float32 happens at load time in the notebook, and the
MobileNetV2 preprocessing (``[0, 255] -> [-1, 1]``) lives inside the model graph
(``src/models/build_model.py``), so arrays must NOT be rescaled here.

The train/val split and the class weights are intentionally left out — they
stay in the notebook so the split remains reproducible with the same
``random_state``.

Run with::

    uv run python -m src.preprocessing.pack_dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    CHANNELS,
    CLASSES,
    DATA_CROPPED_PATH,
    IMG_SIZE,
    LANDMARK_CSV_PATH,
)

# Output file holding the packed arrays (images, angles, labels).
PACKED_DATASET_PATH = "data/cropped/focusai_data.npz"


def load_images(data_dir: Path) -> dict[tuple[str, str], tuple[np.ndarray, int]]:
    """
    Load every cropped face image keyed by its (class, filename) pair.

    Reads each ``.jpg`` from ``<data_dir>/<class>/`` in grayscale, resizes it to
    ``IMG_SIZE`` and reshapes it to ``(*IMG_SIZE, CHANNELS)``. Pixels are kept as
    raw ``uint8`` in ``[0, 255]`` (OpenCV's native dtype) — the cast to float32
    happens at load time in the notebook and rescaling to ``[-1, 1]`` happens
    inside the model graph, so the stored arrays stay raw and integer.

    Args:
        data_dir: Path to ``data/cropped`` containing the class subfolders.

    Returns:
        Mapping ``(class_name, filename) -> (image_array, label_index)`` where
        ``image_array`` is ``uint8`` and ``label_index`` is the position of the
        class in ``CLASSES`` (focused=0, distracted=1).
    """
    image_store: dict[tuple[str, str], tuple[np.ndarray, int]] = {}

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        jpg_files = sorted(class_dir.glob("*.jpg"))
        print(f"{class_name}: {len(jpg_files)} images")
        for img_path in jpg_files:
            # cv2.imread + resize already yield uint8 [0, 255]; keep it that way
            # to halve/quarter the .npz size. The float32 cast happens in the
            # notebook and rescaling to [-1, 1] happens inside the model graph.
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img_arr = img.reshape(*IMG_SIZE, CHANNELS)
            image_store[(class_name, img_path.name)] = (img_arr, label_idx)

    print(f"\nTotal images loaded: {len(image_store)}")
    return image_store


def load_angles(csv_path: Path) -> pd.DataFrame:
    """
    Load the head-pose angles CSV, replacing missing angles with the sentinel.

    Null ``yaw``/``pitch``/``roll`` values mean the face was not detected in that
    frame; they are filled with the sentinel ``999.0`` (same convention as the
    notebook and the model's angle branch).

    Args:
        csv_path: Path to ``landmarks.csv``.

    Returns:
        DataFrame with columns ``filename, class, yaw, pitch, roll,
        face_detected`` and no NaN angles.
    """
    angles_df = pd.read_csv(csv_path)

    # NaN angles mean the face was not detected in that frame -> sentinel 999.0
    angles_df[["yaw", "pitch", "roll"]] = angles_df[["yaw", "pitch", "roll"]].fillna(
        999.0
    )

    no_face_count = int((angles_df["yaw"] == 999.0).sum())
    print(f"CSV rows     : {len(angles_df)}")
    print(f"No-face rows : {no_face_count} (sentinel 999.0 applied)")
    return angles_df


def build_arrays(
    image_store: dict[tuple[str, str], tuple[np.ndarray, int]],
    angles_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Join images and angles into aligned numpy arrays.

    Iterates over the CSV rows (preserving CSV order) and joins each row to its
    image via the composite key ``(class, Path(filename).name)``. Rows whose
    image is missing on disk are counted as unmatched and skipped.

    Args:
        image_store: Mapping returned by :func:`load_images`.
        angles_df: DataFrame returned by :func:`load_angles`.

    Returns:
        Tuple ``(X_images, X_angles, y, unmatched)`` where ``X_images`` is
        uint8 ``(n, *IMG_SIZE, CHANNELS)``, ``X_angles`` is float32
        ``(n, 3)``, ``y`` is int32 ``(n,)`` (focused=0, distracted=1), and
        ``unmatched`` is the count of CSV rows without a matching image.
    """
    X_images, X_angles, y_labels = [], [], []
    unmatched = 0

    for _, row in angles_df.iterrows():
        key = (row["class"], Path(row["filename"]).name)
        if key not in image_store:
            # CSV entry has no corresponding image file on disk
            unmatched += 1
            continue
        img_arr, label_idx = image_store[key]
        X_images.append(img_arr)
        X_angles.append([row["yaw"], row["pitch"], row["roll"]])
        y_labels.append(label_idx)

    X_images = np.array(X_images, dtype=np.uint8)
    X_angles = np.array(X_angles, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    return X_images, X_angles, y, unmatched


def main() -> None:
    """
    Pack the cropped dataset into ``data/cropped/focusai_data.npz``.

    Loads images and angles, aligns them, prints per-class counts plus the
    number of unmatched CSV rows, and writes the compressed archive with keys
    ``images``, ``angles`` and ``labels``.
    """
    data_dir = PROJECT_ROOT / DATA_CROPPED_PATH
    csv_path = PROJECT_ROOT / LANDMARK_CSV_PATH
    out_path = PROJECT_ROOT / PACKED_DATASET_PATH

    image_store = load_images(data_dir)
    angles_df = load_angles(csv_path)
    X_images, X_angles, y, unmatched = build_arrays(image_store, angles_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=X_images, angles=X_angles, labels=y)

    print(f"\nAligned samples : {len(y)} | Unmatched : {unmatched}")
    for label_idx, class_name in enumerate(CLASSES):
        count = int((y == label_idx).sum())
        print(f"  {class_name} (label {label_idx}): {count}")
    print(f"\nSaved packed dataset to: {out_path}")
    print(f"  images : {X_images.shape} {X_images.dtype}")
    print(f"  angles : {X_angles.shape} {X_angles.dtype}")
    print(f"  labels : {y.shape} {y.dtype}")


if __name__ == "__main__":
    main()
