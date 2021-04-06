import os
import torch

from lib.utils.common import mkdir_p
from config import BASE_PATH, SAVE_FOLDER


def save_model(model, epoch):
    model_path = os.path.join(BASE_PATH, SAVE_FOLDER, f"epoch_{epoch}.pt")
    model_dir_path = os.path.dirname(model_path)
    mkdir_p(model_dir_path)

    torch.save(model, model_path)


def load_model(path):
    model = torch.load(path)
    model.eval()

    return model
