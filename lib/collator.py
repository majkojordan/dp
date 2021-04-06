import torch
from torch.nn.utils.rnn import pad_sequence


class Collator(object):
    def __init__(
        self,
        device=torch.device("cpu"),
        modify_sessions=False,
    ):
        self.modify_sessions = modify_sessions
        self.device = device

    def __call__(self, session_data):
        transformed_sessions = []
        for modified_session, original_session, session_id in session_data:
            session = modified_session if self.modify_sessions else original_session

            input = session[:-1]

            input = torch.tensor(input, dtype=torch.long)
            label = torch.tensor(session[-1])
            metadata = (session_id, len(input))
            transformed_sessions.append((input, label, metadata))

        inputs, labels, metadata = zip(*transformed_sessions)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=0).to(self.device)
        labels = torch.stack(labels)

        return inputs, labels, metadata
