import os
from dotenv import load_dotenv

load_dotenv()


def load_bool(name, default=False):
    if name in os.environ:
        return bool(os.getenv(name) != "False")
    return default


DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
BASE_PATH = os.getenv("BASE_PATH") or "."
DATA_DIR = os.getenv("DATA_DIR") or "data"
DATASET = os.getenv("DATASET")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR") or "checkpoints"
DEBUG_FOLDER = os.getenv("DEBUG_FOLDER") or "debug"
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS") or 10)
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE") or 100)
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE") or 100)
NUM_LAYERS = int(os.getenv("NUM_LAYERS") or 1)
LEARNING_RATE = float(os.getenv("LEARNING_RATE") or 0.001)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_TRESHOLD") or 0)
INPUT_DROPOUT = float(os.getenv("INPUT_DROPOUT") or 0.2)
HIDDEN_DROPOUT = float(os.getenv("HIDDEN_DROPOUT") or 0.5)
DEBUG = load_bool("DEBUG")
SAVE_CHECKPOINTS = load_bool("SAVE_CHECKPOINTS")
USE_CATEGORY_SIMILARITY = load_bool("USE_CATEGORY_SIMILARITY")