import os
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
BASE_PATH = os.getenv("BASE_PATH") or "."
DATA_DIR = os.getenv("DATA_DIR") or "data"
DATASET = os.getenv("DATASET")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR") or "checkpoints"
SAVE_CHECKPOINTS = os.getenv("SAVE_CHECKPOINTS") or False
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS") or 10)
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE") or 100)
NUM_LAYERS = int(os.getenv("NUM_LAYERS") or 1)
LEARNING_RATE = float(os.getenv("LEARNING_RATE") or 0.001)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_TRESHOLD") or 0)