import numpy as np

from config import SIMILARITY_THRESHOLD, USE_CATEGORY_SIMILARITY


class PreferenceChange:
    def __init__(self, dataset):
        self.dataset = dataset

    def split_session(self, session):
        # removes unrelated old events from session - keeps only last relevant subsession
        item_similarity = [
            self.dataset.wv_category_model.wv.similarity(
                self.dataset.idx_to_category[session[i]],
                self.dataset.idx_to_category[session[i + 1]],
            )
            if USE_CATEGORY_SIMILARITY
            else self.dataset.wv_model.wv.similarity(
                self.dataset.idx_to_item[session[i]],
                self.dataset.idx_to_item[session[i + 1]],
            )
            for i in range(len(session) - 1)
        ]
        # idx + 1, as similarity is between pairs, so similarities are shifted by one
        split_indexes = [
            idx + 1
            for idx, similarity in enumerate(item_similarity)
            if similarity < SIMILARITY_THRESHOLD
        ]
        last_subsession = np.split(session, split_indexes)[-1]
        return last_subsession

    def filter_session(self, session):
        if USE_CATEGORY_SIMILARITY:
            last_interaction = self.dataset.idx_to_category[session[-1]]
            filtered_session = [
                i
                for i in session
                if self.dataset.wv_category_model.wv.similarity(
                    self.dataset.idx_to_category[i], last_interaction
                )
                > SIMILARITY_THRESHOLD
            ]
        else:
            last_interaction = self.dataset.idx_to_item[session[-1]]
            filtered_session = [
                i
                for i in session
                if self.dataset.wv_model.wv.similarity(
                    self.dataset.idx_to_item[i], last_interaction
                )
                > SIMILARITY_THRESHOLD
            ]

        return filtered_session