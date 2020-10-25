import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import timedelta
from nn import RNN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm, trange
from torch.utils.data import random_split

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


# load data
dataset = SequenceDataset()
test_size = min(int(0.2 * len(dataset)), 10000)
train_size = len(dataset) - test_size
train, test = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
print(f"Train size: {len(train)} sessions, test size: {len(test)} sessions")

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(f"Running on {device_name}")


# create model
model = RNN(
    input_size=dataset.item_count,
    output_size=dataset.item_count,
    hidden_size=100,
    batch_size=BATCH_SIZE,
    device=device,
)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# train
epochs = 2
print(f"Training")
for i in range(epochs):
    print(f"Epoch: {i + 1} / {epochs}")
    hits = 0
    total = 0
    for batch, labels in tqdm(train_loader):
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        model.reset_hidden()

        y_pred = model(batch)

        label_indexes = torch.argmax(labels, axis=1)
        loss = loss_function(y_pred, label_indexes)
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


# evaluate
print("Evaluation")
model.eval()

with torch.no_grad():
    hits = 0
    popular_hits = 0
    total_count = 0
    for batch, labels in tqdm(test_loader):
        batch = batch.to(device)
        labels = labels.to(device)

        model.reset_hidden()

        y_pred = model(batch)

        predicted_indexes = torch.topk(y_pred, 10, axis=1).indices
        label_indexes = torch.argmax(labels, axis=1)

        count = batch.batch_sizes[0].item()
        for i in range(count):
            hits += int(label_indexes[i] in predicted_indexes[i])
            popular_hits += int(
                label_indexes[i].cpu() in dataset.most_popular_item_indexes
            )

        total_count += count

    acc = (hits / total_count) * 100
    popular_acc = (popular_hits / total_count) * 100
    print(f"acc@10: {acc:.4f}%, popular acc@10: {popular_acc:.4f}%")
