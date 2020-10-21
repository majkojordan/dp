import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import DB_CONNECTION_STRING

class SequenceDataset(Dataset):
    def __init__(self):
        df = pd.read_sql(
            "SELECT * FROM preprocessed_events_month_1603127051", DB_CONNECTION_STRING
        )

        item_indexes = (
            df.groupby("product_id")["product_id"].first().reset_index(level=0, drop=True)
        )
        # item_indexes = df.groupby("product_id")["product_id"].first()
        # item_indexes = df["product_id"].unique()
        s = df.groupby("session_id")["product_id"].apply(list)
        s = s[s.apply(lambda x: len(x) < 100)]
        s = [(torch.tensor(x[:-1]), torch.tensor(x[-1])) for x in s.tolist()]
        (inputs, labels) = zip(*s)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

        self.s = list(zip(inputs, labels))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return self.s[idx]
