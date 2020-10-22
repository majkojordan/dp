import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import DB_CONNECTION_STRING

class SequenceDataset(Dataset):
    def __init__(self):
        df = pd.read_sql(
            "SELECT * FROM preprocessed_events_month_1603127051 limit 100", DB_CONNECTION_STRING
        )

        self.item_mapping = (
            df.groupby("product_id")["product_id"].first().reset_index(level=0, drop=True)
        )
        self.item_count = self.item_mapping.size

        sessions = df.groupby("session_id")["product_id"].apply(list)
        sessions = sessions[sessions.apply(lambda x: len(x) < 100)]
        sessions = sessions.apply(lambda s: [self.one_hot_encode(i) for i in s])
        # TODO - more efficient one hot encoding - only before feeding to model to not waste memory - check pytorch scatter
        sessions = [(torch.tensor(x[:-1]), torch.tensor(x[-1])) for x in sessions.tolist()]
        (inputs, labels) = zip(*sessions)
        # lengths = [len(x) for x in inputs]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        # TODO - pack sequences
        # print(torch.nn.utils.rnn.pack_padded_sequence(inputs[:3], lengths[:3], batch_first=True, enforce_sorted=False))

        self.sessions = list(zip(inputs, labels))

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return self.sessions[idx]
    

    def one_hot_encode(self, item):
        vector = torch.zeros(self.item_count)
        vector[self.item_mapping[self.item_mapping == item].index] = 1
        return vector.tolist()


    def one_hot_decode(self, vector):
        item_index = torch.argmax(vector).item()
        return self.item_mapping.iloc[item_index]
