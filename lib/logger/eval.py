import os
import atexit
import pandas as pd

from config import BASE_PATH, EVAL_FOLDER
from lib.utils.common import mkdir_p


class EvalLogger:
    def __init__(self, epoch):
        file_path = os.path.join(BASE_PATH, EVAL_FOLDER, f"epoch_{epoch}.xlsx")
        eval_dir_path = os.path.dirname(file_path)
        mkdir_p(eval_dir_path)

        self.writer = pd.ExcelWriter(file_path, engine="openpyxl")
        self.data = {}

        atexit.register(self.cleanup)

    def log(self, session_ids, predicted_indexes, labels):
        for i in range(len(session_ids)):
            is_hit = labels[i] in predicted_indexes[i]
            self.data[session_ids[i]] = is_hit

    def write(self, sheet_name):
        pd.Series(self.data).to_excel(
            self.writer, sheet_name=sheet_name, header=["hit"]
        )
        self.writer.save()

    def reset(self):
        self.data = {}

    def cleanup(self):
        self.writer.close()