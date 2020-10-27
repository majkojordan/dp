import os
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

DATASET_PATH = os.getenv("DATASET_PATH")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
EPOCHS = int(os.getenv("EPOCHS") or 10)
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE") or 100)
NUM_LAYERS = int(os.getenv("NUM_LAYERS") or 1)
LEARNING_RATE = float(os.getenv("LEARNING_RATE") or 0.001)