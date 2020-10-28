import torch
import os

from datetime import datetime
from pathlib import Path

from config import BASE_PATH, CHECKPOINT_DIR


def print_line_separator():
    print("-" * 72, "\n")


def get_timestamp():
    return str(int(datetime.now(tz=None).timestamp()))


def mkdir_p(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(dir, model, optimizer, epoch, loss):
    path = os.path.join(BASE_PATH, CHECKPOINT_DIR, dir, f"epoch_{epoch}.tar")
    dir_path = os.path.dirname(path)

    mkdir_p(dir_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )

    return path


def load_checkpoint(dir, epoch, model, optimizer):
    path = os.path.join(BASE_PATH, CHECKPOINT_DIR, dir, f"epoch_{epoch}.tar")
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint