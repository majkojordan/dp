import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SequenceDataset(Dataset):
    def __init__(self, dataset_path):
        events = pd.read_csv(dataset_path)

        self.item_mapping = (
            events.groupby("product_id")["product_id"]
            .first()
            .reset_index(level=0, drop=True)
            .rename_axis("product_idx")
            .reset_index()  # add new index as column
        )

        self.item_count = self.item_mapping.size
        events = events.join(self.item_mapping.set_index("product_id"), on="product_id")

        sessions = events.groupby("session_id")["product_idx"].apply(list)
        sessions = sessions[sessions.apply(lambda x: len(x) > 2 and len(x) < 100)]

        sessions = [
            (torch.tensor(s[:-1]), torch.tensor(s[-1]), len(s) - 1) for s in sessions
        ]

        self.inputs, self.labels, self.lengths = zip(*sessions)

        most_popular_items = (
            events.groupby(["customer_id", "product_idx"])
            .agg({"customer_id": "first", "product_idx": "first"})
            .reset_index(drop=True)
        )
        self.most_popular_items = (
            most_popular_items["product_idx"].value_counts().head(10).keys().tolist()
        )

        print(
            f"Dataset initialized ({len(self.inputs)} sessions, {self.item_count} items)"
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.lengths[idx]

    def item_to_idx(self, item):
        return self.item_mapping[self.item_mapping == item].index[0]

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
