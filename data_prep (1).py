"""
v2 data prep: extract 68 dlib landmarks + left/right eye crops for every row
in tracking_data, writing three new columns (landmarks, left_eye, right_eye).

Run from the project root:
    uv run python src/v2/data_prep.py                 # auto-parallel
    uv run python src/v2/data_prep.py --workers 1     # single-process (debug)
    uv run python src/v2/data_prep.py --workers 8
"""
import argparse
import io
import multiprocessing as mp
import os
import sqlite3
import time

import cv2
import dlib
import numpy as np
from skimage import io as skio


DB_PATH = "data/eye_track.db"
PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"

EYE_CROP_SIZE = 64       # output eye crop is EYE_CROP_SIZE x EYE_CROP_SIZE grayscale
EYE_MARGIN = 0.3         # extra padding around the eye bbox (fraction of bbox side)
COMMIT_EVERY = 10
CHUNKSIZE = 8            # tasks per worker batch in imap_unordered

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))


def ensure_schema(cursor):
    """Add v2 columns if they don't already exist. Idempotent."""
    for col, coltype in [
        ("landmarks", "BLOB"),
        ("left_eye", "BLOB"),
        ("right_eye", "BLOB"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE tracking_data ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass


def get_landmarks(image, detector, predictor):
    """Return (68, 2) float32 landmarks for the largest face, or None.

    Fast path: dlib detection with upsample=0. If that finds no face we fall
    back to upsample=1 (the slower, more sensitive setting the old pipeline
    used), so detection quality is never worse than before.
    """
    dets = detector(image, 0)
    if len(dets) == 0:
        dets = detector(image, 1)
    if len(dets) == 0:
        return None
    det = max(dets, key=lambda d: d.width() * d.height())
    shape = predictor(image, det)
    lm = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        lm[i] = (shape.part(i).x, shape.part(i).y)
    return lm


def crop_eye(image, eye_points, out_size=EYE_CROP_SIZE, margin=EYE_MARGIN):
    """Square crop around `eye_points` with margin, resized to grayscale out_size x out_size."""
    x_min, y_min = eye_points.min(axis=0)
    x_max, y_max = eye_points.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    side = max(x_max - x_min, y_max - y_min) * (1.0 + 2.0 * margin)
    half = side / 2.0

    h, w = image.shape[:2]
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def encode_png(gray):
    """Encode a grayscale uint8 array as PNG bytes (lossless)."""
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Worker (used by the multiprocessing pool; each child holds its own dlib
# models and its own read-only SQLite connection).
# ---------------------------------------------------------------------------

_detector = None
_predictor = None
_read_conn = None


def _worker_init():
    global _detector, _predictor, _read_conn
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor(PREDICTOR_PATH)
    _read_conn = sqlite3.connect(DB_PATH)


def _process_row(record_id):
    """Fetch the image for `record_id`, compute v2 features, return blobs.

    Returns (record_id, landmarks_bytes, left_png, right_png, error_or_None).
    """
    try:
        cur = _read_conn.cursor()
        cur.execute("SELECT image FROM tracking_data WHERE id = ?", (record_id,))
        row = cur.fetchone()
        if row is None:
            return (record_id, None, None, None, "row missing")

        image = skio.imread(io.BytesIO(row[0]))

        lm = get_landmarks(image, _detector, _predictor)
        if lm is None:
            return (record_id, None, None, None, "no face")

        left = crop_eye(image, lm[LEFT_EYE_IDX])
        right = crop_eye(image, lm[RIGHT_EYE_IDX])
        if left is None or right is None:
            return (record_id, None, None, None, "eye crop failed")

        return (record_id, lm.tobytes(), encode_png(left), encode_png(right), None)
    except Exception as e:
        return (record_id, None, None, None, f"error[{type(e).__name__}]: {e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def prepare_data(num_workers):
    t_start = time.time()

    with sqlite3.connect(DB_PATH) as conn:
        ensure_schema(conn.cursor())
        conn.commit()

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM tracking_data WHERE landmarks IS NULL")
        ids = [r[0] for r in cur.fetchall()]

    total = len(ids)
    print(f"Processing {total} records missing v2 features... (workers={num_workers})")
    if total == 0:
        print("Nothing to do.")
        return

    processed = 0
    failed = 0
    fail_reasons: dict[str, int] = {}

    write_conn = sqlite3.connect(DB_PATH)
    write_cursor = write_conn.cursor()

    pool = None
    try:
        if num_workers <= 1:
            _worker_init()
            results_iter = (_process_row(rid) for rid in ids)
        else:
            pool = mp.Pool(processes=num_workers, initializer=_worker_init)
            results_iter = pool.imap_unordered(_process_row, ids, chunksize=CHUNKSIZE)

        for idx, (record_id, lm_bytes, left_png, right_png, err) in enumerate(
            results_iter, 1
        ):
            if err is not None:
                failed += 1
                reason = err.split(":", 1)[0] if err.startswith("error") else err
                if failed <= 5 and err.startswith("error"):
                    print(f"    first-failures sample (id={record_id}): {err}")
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            else:
                write_cursor.execute(
                    "UPDATE tracking_data "
                    "SET landmarks = ?, left_eye = ?, right_eye = ? WHERE id = ?",
                    (lm_bytes, left_png, right_png, record_id),
                )
                processed += 1

            if idx % COMMIT_EVERY == 0:
                write_conn.commit()
                rate = idx / max(1e-6, time.time() - t_start)
                breakdown = (
                    " [" + ", ".join(f"{k}={v}" for k, v in sorted(fail_reasons.items())) + "]"
                    if fail_reasons else ""
                )
                print(
                    f"  [{idx}/{total}] processed={processed} failed={failed}"
                    f"{breakdown} ({rate:.1f} rec/s)"
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        write_conn.commit()
        write_conn.close()

    elapsed = time.time() - t_start
    rate = total / max(1e-6, elapsed)
    print("=" * 50)
    print(f"Done in {elapsed:.1f}s  |  processed={processed}  failed={failed}  "
          f"({rate:.1f} rec/s)")
    if fail_reasons:
        print("Failure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print("=" * 50)


def main():
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"parallel worker processes (default: {default_workers})")
    args = parser.parse_args()
    prepare_data(args.workers)


if __name__ == "__main__":
    main()
