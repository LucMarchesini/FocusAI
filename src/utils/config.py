IMG_SIZE = (224, 224)

CHANNELS = 1

CLASSES = ["focused", "distracted"]

DATA_RAW_PATH = "data"

DATA_CROPPED_PATH = "data/cropped"

MODEL_PATH = "models/focus_model.h5"

LANDMARK_CSV_PATH = "data/cropped/landmarks.csv"

NO_FACE_LOG_PATH = "data/cropped/no_face_log.txt"

MEDIAPIPE_MODEL_PATH = "models/face_landmarker.task"

MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
