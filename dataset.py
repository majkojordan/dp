import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec

from config import SIMILARITY_THRESHOLD
from preprocess import remove_unfrequent_items


class SequenceDataset(Dataset):
    def __init__(self, dataset_path):
        events = pd.read_csv(dataset_path)

        # remove items that don't have significant impact
        events = remove_unfrequent_items(events, 5)

        # create item-index mapping
        self.item_mapping = (
            events.groupby("product_id")["product_id"]
            .first()
            .reset_index(level=0, drop=True)
            .rename_axis("product_idx")
            .reset_index()  # add new index as column
        )
        self.item_count = self.item_mapping.size

        # add item indexes to events
        events = events.join(self.item_mapping.set_index("product_id"), on="product_id")
        # transform to strings so it can be used with word2vec
        events["product_id"] = events["product_id"].astype(str)
        # sessions = events.groupby("session_id")["product_idx"].apply(list)
        sessions = (
            events.groupby("session_id")["product_id"]
            .apply(list)
            .rename("session_events")
        )

        # remove one item sessions
        sessions = sessions[sessions.apply(lambda x: len(x) > 2)]
        # remove sessions where label and last item are the same
        # sessions = sessions[sessions.apply(lambda x: x[-1] != x[-2])]
        # remove sessions where label is in input sequence
        sessions = sessions[sessions.apply(lambda x: x[-1] not in x[:-1])]

        # create word2vec embeddings
        word_model = Word2Vec(sessions.tolist(), size=100, window=5, min_count=1)
        word_model.init_sims(replace=True)
        self.word_model = word_model

        # create mapping dictionaries
        self.idx_to_item = word_model.wv.index2word
        self.item_to_idx = {item: idx for idx, item in enumerate(self.idx_to_item)}

        # ensure that sessions consist only from items that are in word2vec dictionary and session lengths are within boundary
        sessions = sessions.apply(lambda x: [i for i in x if i in word_model.wv])

        # remember long session ids - they are used in evaluation
        self.long_session_ids = sessions[
            sessions.apply(lambda x: len(x) > 10)
        ].index.tolist()
        # print(long_session_ids)
        # split sessions to subsessions
        # print(sum(len(x) > 10 for x in sessions.tolist()) / len(sessions.tolist()))
        if SIMILARITY_THRESHOLD > 0:
            self.similarity_treshold = SIMILARITY_THRESHOLD
            sessions = sessions.apply(self.split_session)

        # print(sessions["02f3e799-c152-4734-a203-bad74e2366e4"])

        # remove one item sessions
        sessions = sessions[sessions.apply(lambda x: len(x) > 2)]

        # map item names to indexes
        sessions = sessions.apply(lambda x: list(map(lambda i: self.item_to_idx[i], x)))

        # convert to input and label tensors + metadata
        sessions = [
            (
                torch.tensor(s[:-1]),
                torch.tensor(s[-1]),
                (session_id, len(s) - 1),
            )
            for s, session_id in zip(sessions, sessions.index)
        ]

        self.inputs, self.labels, self.metadata = zip(*sessions)

        # precompute most popular items
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
        return self.inputs[idx], self.labels[idx], self.metadata[idx]

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

    def split_session(self, session):
        # removes unrelated old events from session
        item_similarity = [
            self.word_model.wv.similarity(session[i], session[i + 1])
            for i in range(len(session) - 1)
        ]
        # idx + 1, as similarity is between pairs, so similarities are shifted by one
        split_indexes = [
            idx + 1
            for idx, similarity in enumerate(item_similarity)
            if similarity < self.similarity_treshold
        ]
        last_subsession = np.split(session, split_indexes)[-1]
        return last_subsession
