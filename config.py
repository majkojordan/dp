import os
from dotenv import load_dotenv

from lib.constants import OFF


load_dotenv()


def load_bool(name, default=False):
    if name in os.environ:
        return bool(os.getenv(name) != "False")
    return default


DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
BASE_PATH = os.getenv("BASE_PATH") or "."
DATA_DIR = os.getenv("DATA_DIR") or "data"
DATASET = os.getenv("DATASET")
DEBUG_FOLDER = os.getenv("DEBUG_FOLDER") or "debug"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS") or 10)
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE") or 100)
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE") or 100)
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE") or 5)
MAX_TEST_SIZE = int(os.getenv("MAX_TEST_SIZE") or 10000)
MAX_VALIDATION_SIZE = int(os.getenv("MAX_TEST_SIZE") or 10000)
NUM_LAYERS = int(os.getenv("NUM_LAYERS") or 1)
MANUAL_SEED = int(os.getenv("MANUAL_SEED") or 0)
DETECT_PREFERENCE_CHANGE = int(
    os.getenv("DETECT_PREFERENCE_CHANGE") or OFF
)  # 0 - off, 1 - split, 2 - filter
LEARNING_RATE = float(os.getenv("LEARNING_RATE") or 0.001)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD") or 0)
INPUT_DROPOUT = float(os.getenv("INPUT_DROPOUT") or 0)
HIDDEN_DROPOUT = float(os.getenv("HIDDEN_DROPOUT") or 0)
DEBUG = load_bool("DEBUG")
USE_CATEGORY_SIMILARITY = load_bool("USE_CATEGORY_SIMILARITY")
EVALUATE_MODEL = load_bool("EVALUATE_MODEL")
EVAL_FOLDER = os.getenv("EVAL_FOLDER") or "evaluation"
SAVE_MODEL = load_bool("SAVE_MODEL")
SAVE_FOLDER = os.getenv("SAVE_FOLDER") or "model"
HYBRID_ORIGINAL_MODEL_PATH = os.getenv("HYBRID_ORIGINAL_MODEL_PATH")
MODIFY_TRAIN = load_bool("MODIFY_TRAIN", True)
