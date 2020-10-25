import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import timedelta
from nn import RNN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from config import BATCH_SIZE, DB_CONNECTION_STRING
from dataset import SequenceDataset


# def one_hot_encode(item):
#     vector = torch.zeros(dataset.item_count)
#     vector[item_indexes[item_indexes == item].index] = 1
#     return vector.tolist()


# def one_hot_decode(vector):
#     item_index = torch.argmax(vector).item()
#     return item_indexes.iloc[item_index]


def load_data(table_name="preprocessed_events", count="1000"):
    query = f"SELECT * FROM {table_name} ORDER BY session_id DESC LIMIT {count}"
    try:
        return pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None


def get_popular_items(table_name="product_counts", count=10):
    query = f"SELECT product_id FROM {table_name} ORDER BY count DESC LIMIT {count}"
    try:
        df = pd.read_sql(query, DB_CONNECTION_STRING)
    except:
        return None

    return df["product_id"].tolist()


def collate_fn(sessions):
    sessions = [[dataset.one_hot_encode(i) for i in s] for s in sessions]
    # # TODO - more efficient one hot encoding - only before feeding to model to not waste memory - check pytorch scatter
    sessions = [(torch.tensor(x[:-1]), torch.tensor(x[-1])) for x in sessions]

    inputs, labels = zip(*sessions)

    input_lengths = [len(x) for x in inputs]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    inputs = pack_padded_sequence(
        inputs, input_lengths, batch_first=True, enforce_sorted=False
    )

    labels = torch.stack(labels)

    return inputs, labels


dataset = SequenceDataset()
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# inputs, labels = next(iter(dataloader))
# print(labels)


# create model
model = RNN(
    input_size=dataset.item_count,
    output_size=dataset.item_count,
    hidden_size=100,
    batch_size=BATCH_SIZE,
)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# train
for i in range(20):
    hits = 0
    total = 0
    for batch, labels in dataloader:
        optimizer.zero_grad()
        model.reset_hidden()

        y_pred = model(batch)

        label_indexes = torch.argmax(labels, axis=1)
        loss = loss_function(y_pred, label_indexes)
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")


# evaluate
# TODO - test and validation datasets
model.eval()

with torch.no_grad():
    total_hits = 0
    total_count = 0
    for batch, labels in dataloader:
        model.reset_hidden()

        y_pred = model(batch)

        # print(
        #     [dataset.one_hot_decode(i) for i in y_pred.tolist()],
        #     [dataset.one_hot_decode(i) for i in labels.tolist()],
        # )

        predicted_indexes = torch.argmax(y_pred, axis=1)
        label_indexes = torch.argmax(labels, axis=1)

        hits = torch.sum(predicted_indexes == label_indexes).item()
        count = batch.batch_sizes[0].item()

        total_hits += hits
        total_count += count

    acc = (total_hits / total_count) * 100
    print(f"acc: {acc:.4f}%")
