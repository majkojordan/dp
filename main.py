import pandas as pd
import torch
import os

from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from datetime import timedelta
from tqdm import tqdm

from nn import RNN
from config import (
    BATCH_SIZE,
    BASE_PATH,
    DB_CONNECTION_STRING,
    DATA_DIR,
    DATASET,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    SAVE_CHECKPOINTS,
)
from dataset import SequenceDataset
from utils import get_timestamp, save_checkpoint, load_checkpoint, print_line_separator


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
dataset_path = os.path.join(BASE_PATH, DATA_DIR, DATASET)
dataset = SequenceDataset(dataset_path)
test_size = min(int(0.2 * len(dataset)), 10000)
train_size = len(dataset) - test_size
train, test = random_split(dataset, [train_size, test_size])
dataloaders = {
    "train": DataLoader(
        train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    ),
    "test": DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    ),
}
data_sizes = {"train": train_size, "test": test_size}

print(f"Train size: {train_size} sessions, test size: {test_size} sessions")


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
print_line_separator()


# load checkpoint
use_checkpoint = False
checkpoint = {"name": "checkpoint_1603809820", "epoch": 2}

if use_checkpoint:
    checkpoint = load_checkpoint(
        checkpoint["name"], checkpoint["epoch"], model, optimizer
    )
    print(checkpoint["loss"].item(), checkpoint["epoch"])


# calculate baseline
print("Calculating baseline scores - most popular")
popular_acc = 0
popular_hits_10 = 0
for _, labels in tqdm(dataloaders["test"]):
    popular_hits = sum([l == dataset.most_popular_items[0] for l in labels.tolist()])
    popular_hits_10 += sum([l in dataset.most_popular_items for l in labels.tolist()])
popular_acc = popular_hits / data_sizes["test"] * 100
popular_acc_10 = popular_hits_10 / data_sizes["test"] * 100
print(f"Baseline - acc@1: {popular_acc:.4f}, acc@10: {popular_acc_10:.4f}\n")
print_line_separator()


# train
def train(dataloaders, epochs=10, save_checkpoints=False):
    print(f"Training")

    save_dir = f"checkpoint_{get_timestamp()}"

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch} / {epochs}\n")
        for phase in ["train", "test"]:
            print(f"Phase: {phase}")

            is_train = phase == "train"

            if is_train:
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(is_train):
                hits = 0
                hits_10 = 0
                running_loss = 0

                for batch, labels in tqdm(dataloaders[phase]):
                    batch = batch.to(device, dtype=torch.float)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    model.reset_hidden()

                    y_pred = model(batch)

                    loss = loss_function(y_pred, labels)

                    curr_batch_size = batch.batch_sizes[0].item()
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    else:
                        # calculate hits@10 - only for test as it would slow down training
                        predicted_indexes_10 = torch.topk(y_pred, 10, axis=1).indices
                        hits_10 += sum(
                            [l in predicted_indexes_10 for l in labels.tolist()]
                        )

                    predicted_indexes = torch.argmax(y_pred, 1)
                    hits += torch.sum(predicted_indexes == labels).item()
                    running_loss += loss.item() * curr_batch_size

            avg_loss = running_loss / data_sizes[phase]
            acc = hits / data_sizes[phase] * 100
            if is_train:
                print(f"Avg. loss: {avg_loss:.8f}, acc@1: {acc:.4f}\n")
            else:
                acc_10 = hits_10 / data_sizes[phase] * 100
                print(
                    f"Avg. loss: {avg_loss:.8f}, acc@1: {acc:.4f}, acc@10: {acc_10:.4f}\n"
                )

        if save_checkpoints:
            save_checkpoint(save_dir, model, optimizer, epoch, loss)

        print_line_separator()


train(dataloaders, epochs=EPOCHS, save_checkpoints=SAVE_CHECKPOINTS)
