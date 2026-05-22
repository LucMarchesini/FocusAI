# CLAUDE.md — FocusAI Coding Reference

## Project
Focus detection system using webcam-based computer vision.
Classifies user state as `focused` (0) or `distracted` (1) in real time.

## Stack
- Python 3.12
- TensorFlow/Keras — model training
- MediaPipe — face detection and landmark extraction
- OpenCV — image processing
- Flask — data collection web app
- pandas, numpy, scikit-learn — data pipeline
- uv — package manager (use `uv add`, never `pip install`)

## Structure
FOCUSAI/
├── collection/          # Flask app for data collection
│   ├── templates/index.html
│   ├── __init__.py
│   └── app.py
├── data/
│   ├── focused/         # raw images, label 0
│   ├── distracted/      # raw images, label 1
│   └── cropped/
│       ├── focused/     # face-cropped images, label 0
│       ├── distracted/  # face-cropped images, label 1
│       ├── landmarks.csv
│       └── no_face_log.txt
├── models/              # saved model files (not tracked by git)
├── notebooks/           # Jupyter notebooks
│   ├── train_cnn.ipynb  # legacy single-input model (do not modify)
│   └── train_dual.ipynb # current dual-input model
├── src/
│   ├── utils/
│   │   └── config.py    # single source of truth for all constants and paths
│   └── preprocessing/
│       └── crop_faces.py # full preprocessing pipeline
├── CLAUDE.md
├── PROGRESS.md
└── pyproject.toml

## Config
All paths and constants live in src/utils/config.py.
Always import from there. Never hardcode paths.
Constants: IMG_SIZE, CHANNELS, CLASSES, DATA_RAW_PATH,
DATA_CROPPED_PATH, MODEL_PATH, LANDMARK_CSV_PATH,
NO_FACE_LOG_PATH, MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_URL

## Path Resolution
- Scripts in src/: PROJECT_ROOT = Path(__file__).resolve().parents[2]
- Notebooks in notebooks/: PROJECT_ROOT = Path("../").resolve()
- Always use pathlib. Never hardcode absolute paths.
- Always add PROJECT_ROOT to sys.path before importing from src/

## Data
- Raw images: data/focused/ and data/distracted/
- Cropped images: data/cropped/focused/ and data/cropped/distracted/
- Image format: grayscale JPEG, 224x224, normalized to [0,1]
- landmarks.csv columns: filename, class, yaw, pitch, roll, face_detected
- Null angles = no face detected → replace with sentinel 999.0 at training time
- Class imbalance: ~997 focused vs ~1339 distracted → always use class weights

## Model Architecture (dual-input)
- Input 1 "image_input": (224, 224, 1) grayscale image
- Input 2 "angle_input": (3,) head pose angles [yaw, pitch, roll]
- Image branch: 3× Conv2D+MaxPooling → Flatten → Dense(128) → Dropout(0.4)
- Angle branch: Dense(16) → Dense(16)
- Fusion: Concatenate → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)
- Output: 0 = focused, 1 = distracted
- Saved to: models/focus_model.h5

## Landmarks CSV alignment
When joining images with angles, always use:
    key = (row["class"], Path(row["filename"]).name)
The filename column stores full paths like data/cropped/focused/img.jpg
The image store uses only the filename img.jpg

## Git conventions
- Language: Portuguese
- Format: type: descrição curta
- Types: feat, fix, chore, refactor, docs
- Examples:
    feat: adicionar função de estimativa de pose da cabeça
    fix: corrigir alinhamento entre imagens e CSV
    chore: atualizar .gitignore

## Rules
- Never modify train_cnn.ipynb
- Never modify data/focused/ or data/distracted/
- Never hardcode absolute paths
- Never use pip install — always uv add
- Always guard scripts with if __name__ == "__main__"
- Always add docstrings and type hints to functions in src/
- Keep cells in notebooks runnable top to bottom
