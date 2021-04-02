import os
import atexit
from pprint import pformat
import numpy as np

from lib.utils.common import mkdir_p
from config import BASE_PATH, DEBUG_FOLDER


class SessionDebugger:
    def __init__(self, epoch):
        debug_path = os.path.join(BASE_PATH, DEBUG_FOLDER, f"epoch_{epoch}.txt")
        debug_dir_path = os.path.dirname(debug_path)
        mkdir_p(debug_dir_path)
        self.debug_f = open(debug_path, "w")

        atexit.register(self.cleanup)

    def cleanup(self):
        self.debug_f.close()

    def log(self, dataset, session_ids, predicted_indexes):
        for session_id, predictions in zip(session_ids, predicted_indexes):
            session = [
                dataset.idx_to_info(i) for i in dataset.sessions.loc[session_id].clicks
            ]
            predictions = [dataset.idx_to_info(i) for i in predictions.tolist()]

            self.debug_f.write(
                (
                    f"INPUT:\n{pformat(session[:-1], width=160)}\n\n"
                    f"LABEL:\n{pformat(session[-1], width=160)}\n\n"
                    f"PREDICTIONS:\n{pformat(predictions, width=160)}\n\n"
                    f"long: {session_id in dataset.long_session_ids}\n"
                    f"correct: {session[-1] in predictions}\n"
                    f"{'-' * 72}\n"
                )
            )
