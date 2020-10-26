import pandas as pd
import torch

from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datetime import timedelta
from tqdm import tqdm

from nn import RNN
from config import (
    BATCH_SIZE,
    DB_CONNECTION_STRING,
    DATASET_PATH,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
)
from dataset import SequenceDataset


def collate_fn(sessions):
    inputs, labels, lengths = zip(*sessions)

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0).to(device)
    inputs = one_hot(inputs, dataset.item_count).to(device)

    inputs = pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False
    ).to(device)

    labels = torch.stack(labels)

    return inputs, labels


# load data
dataset = SequenceDataset(DATASET_PATH)
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


# select device
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(f"Running on {device_name}")


# create model
model = RNN(
    input_size=dataset.item_count,
    output_size=dataset.item_count,
    hidden_size=HIDDEN_SIZE,
    batch_size=BATCH_SIZE,
    num_layers=NUM_LAYERS,
    device=device,
)
loss_function = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# evaluate
def test(dataloader):
    print("Evaluation")
    model.eval()

    with torch.no_grad():
        hits = 0
        popular_hits = 0
        total_count = 0
        for batch, labels in tqdm(dataloader):
            batch = batch.to(device, dtype=torch.float)
            labels = labels.to(device)

            model.reset_hidden()

            y_pred = model(batch)

            predicted_indexes = torch.topk(y_pred, 10, axis=1).indices

            count = batch.batch_sizes[0].item()
            for i in range(count):
                hits += int(labels[i] in predicted_indexes[i])
                popular_hits += int(labels[i].cpu() in dataset.most_popular_items)

            total_count += count

        acc = (hits / total_count) * 100
        popular_acc = (popular_hits / total_count) * 100
        print(f"acc@10: {acc:.4f}%, popular acc@10: {popular_acc:.4f}%")

    model.train()


# train
def train(dataloader, epochs=10):
    print(f"Training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1} / {epochs}")
        hits = 0
        total = 0
        for i, (batch, labels) in enumerate(tqdm(train_loader)):
            batch = batch.to(device, dtype=torch.float)
            labels = labels.to(device)

            optimizer.zero_grad()
            model.reset_hidden()

            y_pred = model(batch)

            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 1:
                print(f"Loss: {loss.item()}")

        test(test_loader)
        print(f"Loss: {loss.item()}")


train(train_loader, epochs=EPOCHS)
