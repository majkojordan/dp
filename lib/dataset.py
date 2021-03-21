import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from gensim.models import Word2Vec

from lib.constants import SPLIT_SESSIONS, FILTER_SESSIONS
from config import EMBEDDING_SIZE, WINDOW_SIZE
from preprocess import remove_unfrequent_items


def trainWord2Vec(series, embedding_size=100, window_size=3):
    model = Word2Vec(
        series.tolist(), size=embedding_size, window=window_size, min_count=1
    )
    model.init_sims(replace=True)
    return model


class SequenceDataset(Dataset):
    def __init__(self, dataset_path):
        events = pd.read_csv(dataset_path)

        # remove items that don't have significant impact
        events = remove_unfrequent_items(events, 5)

        # transform to strings so it can be used with word2vec
        events["product_id"] = events["product_id"].astype(str)

        # sort by timestamp
        events = events.sort_values(by=["timestamp"])

        # create sessions
        sessions = (
            events.groupby("session_id", sort=False)
            .agg(
                {
                    "product_id": lambda x: list(x),
                    "customer_id": "first",
                    "timestamp": "first",
                }
            )
            .rename(columns={"product_id": "clicks"})
        )

        # remove one item sessions
        sessions = sessions[sessions["clicks"].apply(lambda x: len(x) > 2)]
        # remove sessions where label is in input sequence
        sessions = sessions[sessions["clicks"].apply(lambda x: x[-1] not in x[:-1])]

        # train word2vec embeddings
        self.wv_model = trainWord2Vec(
            sessions["clicks"], embedding_size=EMBEDDING_SIZE, window_size=WINDOW_SIZE
        )

        # ensure that sessions consist only from items that are in word2vec dictionary
        sessions["clicks"] = sessions["clicks"].apply(
            lambda x: [i for i in x if i in self.wv_model.wv]
        )

        # create mappings
        self.idx_to_item = self.wv_model.wv.index2word
        self.item_to_idx = {item: idx for idx, item in enumerate(self.idx_to_item)}
        self.item_to_title = events.groupby("product_id")["title"].first()
        self.idx_to_title = [self.item_to_title[i] for i in self.idx_to_item]
        self.item_to_category = events.groupby("product_id")["categories"].first()
        self.idx_to_category = [self.item_to_category[i] for i in self.idx_to_item]

        self.item_count = len(self.idx_to_item)

        # get category click sequences
        events["categories"] = events["categories"].astype(str)
        category_sequences = events.groupby("session_id")["categories"].apply(list)

        # remove one item sessions
        category_sequences = category_sequences[
            category_sequences.apply(lambda x: len(x) > 2)
        ]

        # create word2vec embeddings
        self.wv_category_model = trainWord2Vec(
            category_sequences,
            embedding_size=EMBEDDING_SIZE,
            window_size=WINDOW_SIZE,
        )

        # ensure that sessions consist only from items that are in word2vec dictionary
        sessions["clicks"] = sessions["clicks"].apply(
            lambda x: [
                i for i in x if self.item_to_category[i] in self.wv_category_model.wv
            ]
        )

        # remove one item sessions
        sessions = sessions[sessions["clicks"].apply(lambda x: len(x) > 2)]

        # remember long session ids - they are used in evaluation
        self.long_session_ids = sessions[
            sessions["clicks"].apply(lambda x: len(x) > 10)
        ].index.tolist()

        # map item names to indexes
        sessions["clicks"] = sessions["clicks"].apply(
            lambda x: list(map(lambda i: self.item_to_idx[i], x))
        )
        sessions["original_clicks"] = sessions["clicks"]

        self.sessions = sessions

        # precompute most popular items
        most_popular_item_ids = (
            events.groupby(["customer_id", "product_id"])
            .agg({"product_id": "first"})["product_id"]
            .value_counts()
            .head(10)
            .keys()
            .tolist()
        )
        self.most_popular_items = [self.item_to_idx[i] for i in most_popular_item_ids]

        print(
            f"Dataset initialized ({len(self.sessions)} sessions, {self.item_count} items)"
        )

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return (
            self.sessions["clicks"][idx],
            self.sessions["original_clicks"][idx],
            self.sessions.index[idx],
        )

    def adapt_user_preference(self, method, session_modifier):
        # adapt sessions to user preference
        sessions = self.sessions
        if method == SPLIT_SESSIONS:
            sessions["clicks"] = sessions["clicks"].apply(
                lambda x: [*session_modifier.split_session(x[:-1]), x[-1]]
            )
        elif method == FILTER_SESSIONS:
            sessions["clicks"] = sessions["clicks"].apply(
                lambda x: [*session_modifier.filter_session(x[:-1]), x[-1]]
            )

        self.modified_session_ids = [
            id
            for idx, id in enumerate(sessions.index)
            if sessions["clicks"][idx] != sessions["original_clicks"][idx]
        ]

        self.sessions = sessions

    def idx_to_info(self, idx):
        return {
            "title": self.idx_to_title[idx],
            "category": self.idx_to_category[idx],
        }

    def get_item_embeddings(self):
        return self.wv_model.wv.vectors