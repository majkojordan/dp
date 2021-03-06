from datetime import datetime
from pathlib import Path


def print_line_separator():
    print("-" * 72, "\n")


def get_timestamp():
    return str(int(datetime.now(tz=None).timestamp()))


def mkdir_p(path):
    Path(path).mkdir(parents=True, exist_ok=True)
