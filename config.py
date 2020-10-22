import os
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = int(os.getenv("BATCH_SIZE")) or 16
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
