import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import timedelta
from nn import RNN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm, trange

from config import BATCH_SIZE, DB_CONNECTION_STRING
from dataset import SequenceDataset

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
epochs = 5
print(f'Starting training: {epochs} epochs')
for i in range(epochs):
    hits = 0
    total = 0
    for batch, labels in tqdm(dataloader):
        optimizer.zero_grad()
        model.reset_hidden()

        y_pred = model(batch)

        label_indexes = torch.argmax(labels, axis=1)
        loss = loss_function(y_pred, label_indexes)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {i + 1} / {epochs}, loss: {loss.item()}")


# evaluate
# TODO - test and validation datasets
model.eval()

with torch.no_grad():
    hits = 0
    popular_hits = 0
    total_count = 0
    for batch, labels in dataloader:
        model.reset_hidden()

        y_pred = model(batch)

        predicted_indexes = torch.topk(y_pred, 10, axis=1).indices
        label_indexes = torch.argmax(labels, axis=1)

        count = batch.batch_sizes[0].item()
        for i in range(count):
            hits += int(label_indexes[i] in predicted_indexes[i])
            popular_hits += int(label_indexes[i] in dataset.most_popular_item_indexes)

        total_count += count

    acc = (hits / total_count) * 100
    popular_acc = (popular_hits / total_count) * 100
    print(f"acc@10: {acc:.4f}%, popular acc@10: {popular_acc:.4f}%")
