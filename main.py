from lib.utils.data import create_data_samples, create_dataloaders
from lib.session_debugger import SessionDebugger
import torch
import os

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BASE_PATH,
    DEBUG,
    DATA_DIR,
    DATASET,
    DETECT_PREFERENCE_CHANGE,
    EPOCHS,
    HIDDEN_SIZE,
    EMBEDDING_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    HIDDEN_DROPOUT,
    INPUT_DROPOUT,
)
from lib.nn import RNN
from lib.dataset import SequenceDataset
from lib.session_modifier import SessionModifier
from lib.utils.common import print_line_separator
from lib.collator import Collator

# select device
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(f"Running on {device_name}")

# load data
dataset_path = os.path.join(BASE_PATH, DATA_DIR, DATASET)
dataset = SequenceDataset(dataset_path)

# trigger user preference adaptation
session_modifier = SessionModifier(dataset)
dataset.adapt_user_preference(DETECT_PREFERENCE_CHANGE, session_modifier)

train_set, test_set, validation_set = create_data_samples(dataset)
dataloaders = create_dataloaders(
    dataset=dataset,
    train_set=train_set,
    test_set=test_set,
    validation_set=test_set,
    device=device,
)

for phase, dataloader in dataloaders.items():
    print(f"{phase} size: {len(dataloader.dataset)} sessions")

# load embeddings
pretrained_embeddings = dataset.get_item_embeddings()

# create model
model = RNN(
    vocab_size=dataset.item_count,
    embedding_size=EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    batch_size=BATCH_SIZE,
    num_layers=NUM_LAYERS,
    input_dropout=INPUT_DROPOUT,
    hidden_dropout=HIDDEN_DROPOUT,
    device=device,
    pretrained_embeddings=pretrained_embeddings,
)
loss_function = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# calculate baseline
print_line_separator()
print("Calculating baseline scores - most popular")
popular_acc = 0
popular_hits_10 = 0
for _, labels, _ in tqdm(dataloaders["validation"]):
    popular_hits = sum([l == dataset.most_popular_items[0] for l in labels.tolist()])
    popular_hits_10 += sum([l in dataset.most_popular_items for l in labels.tolist()])
validation_size = len(dataloaders["validation"].dataset)
popular_acc = popular_hits / validation_size * 100
popular_acc_10 = popular_hits_10 / validation_size * 100
print(f"Baseline - acc@1: {popular_acc:.4f}, acc@10: {popular_acc_10:.4f}\n")
print_line_separator()


# train
def train(dataloaders, epochs=10, debug=False):
    print(f"Training")

    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch} / {epochs}\n")

        if debug:
            debugger = SessionDebugger(epoch)

        for phase, dataloader in dataloaders.items():
            print(f"Phase: {phase}")

            is_train = phase == "train"

            if is_train:
                model.train()
            else:
                model.eval()

            with torch.set_grad_enabled(is_train):
                hits = 0
                hits_10 = 0
                long_hits_10 = 0
                long_session_count = 0
                running_loss = 0

                for inputs, labels, metadata in tqdm(dataloader):
                    inputs = inputs.to(device, dtype=torch.float)
                    labels = labels.to(device)
                    session_ids, lengths = zip(*metadata)

                    optimizer.zero_grad()
                    model.reset_hidden()

                    y_pred = model(inputs, lengths)

                    loss = loss_function(y_pred, labels)

                    curr_batch_size = inputs.shape[0]
                    if is_train:
                        loss.backward()
                        optimizer.step()
                    else:
                        # calculate hits@10 - only for test as it would slow down training
                        predicted_indexes_10 = torch.topk(y_pred, 10, axis=1).indices
                        hits_10 += sum(
                            [
                                l in predicted_indexes_10[i]
                                for i, l in enumerate(labels.tolist())
                            ]
                        )

                        # show the predictions
                        if debug:
                            debugger.log(
                                dataset=dataset,
                                session_ids=session_ids,
                                predicted_indexes=predicted_indexes_10,
                            )

                        # calculate hits@10 for long sessions only
                        long_indexes = [
                            i in dataset.long_session_ids for i in session_ids
                        ]
                        long_inputs = inputs[long_indexes]
                        long_labels = labels[long_indexes]
                        long_preds = y_pred[long_indexes]

                        predicted_indexes_10 = torch.topk(
                            long_preds, 10, axis=1
                        ).indices
                        long_hits_10 += sum(
                            [
                                l in predicted_indexes_10[i]
                                for i, l in enumerate(long_labels.tolist())
                            ]
                        )
                        long_session_count += long_inputs.shape[0]

                    predicted_indexes = torch.argmax(y_pred, 1)
                    hits += torch.sum(predicted_indexes == labels).item()
                    running_loss += loss.item() * curr_batch_size

            phase_size = len(dataloader.dataset)
            if phase_size == 0:
                print("No sessions")
                continue
            avg_loss = running_loss / phase_size
            acc = hits / phase_size * 100
            if is_train:
                print(f"Avg. loss: {avg_loss:.8f}, acc@1: {acc:.4f}\n")
            else:
                acc_10 = hits_10 / phase_size * 100
                long_acc_10 = (
                    long_hits_10 / long_session_count * 100
                    if long_session_count > 0
                    else 999
                )
                print(
                    f"Avg. loss: {avg_loss:.8f}, acc@1: {acc:.4f}, acc@10: {acc_10:.4f}, long acc@10: {long_acc_10:.4f}\n"
                )

        print_line_separator()


train(dataloaders, epochs=EPOCHS, debug=DEBUG)
