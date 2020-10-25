import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import DB_CONNECTION_STRING


class SequenceDataset(Dataset):
    def __init__(self):
        df = pd.read_sql(
            "SELECT * FROM preprocessed_events_month_1603127051 limit 10000",
            DB_CONNECTION_STRING,
        )

        self.item_mapping = (
            df.groupby("product_id")["product_id"]
            .first()
            .reset_index(level=0, drop=True)
        )
        self.item_count = self.item_mapping.size

        sessions = df.groupby("session_id")["product_id"].apply(list)
        sessions = sessions[sessions.apply(lambda x: len(x) > 2 and len(x) < 100)]

        self.sessions = sessions

        most_popular_items = (
            df.groupby(["customer_id", "product_id"])
            .agg({"customer_id": "first", "product_id": "first"})
            .reset_index(drop=True)
        )
        most_popular_items = (
            most_popular_items["product_id"].value_counts().head(10).keys().tolist()
        )
        self.most_popular_item_indexes = [
            self.item_to_idx(i) for i in most_popular_items
        ]

        print("Dataset initialized")

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return self.sessions.iloc[idx]

    def item_to_idx(self, item):
        return self.item_mapping[self.item_mapping == item].index

    def idx_to_item(self, idx):
        return self.item_mapping.iloc[idx]

    def one_hot_encode(self, item):
        vector = torch.zeros(self.item_count)
        idx = self.item_to_idx(item)
        vector[idx] = 1
        return vector.tolist()

    def one_hot_decode(self, vector):
        idx = torch.argmax(torch.tensor(vector)).item()
        return idx_to_item(idx)
