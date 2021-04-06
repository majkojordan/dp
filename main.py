import os
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BASE_PATH,
    DEBUG,
    DATA_DIR,
    DATASET,
    DETECT_PREFERENCE_CHANGE,
    EPOCHS,
    EVALUATE_MODEL,
    HIDDEN_SIZE,
    EMBEDDING_SIZE,
    HYBRID_ORIGINAL_MODEL_PATH,
    LEARNING_RATE,
    NUM_LAYERS,
    HIDDEN_DROPOUT,
    INPUT_DROPOUT,
    SAVE_MODEL,
)
from lib.utils.data import create_data_samples, create_dataloaders
from lib.utils.model import load_model
from lib.nn import RNN
from lib.dataset import SequenceDataset
from lib.session_modifier import SessionModifier
from lib.utils.common import print_line_separator
from lib.trainer import Trainer

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
    validation_set=validation_set,
    device=device,
    modify_train=True,
    evaluate=EVALUATE_MODEL,
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
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# hybrid
original_model = (
    load_model(HYBRID_ORIGINAL_MODEL_PATH) if HYBRID_ORIGINAL_MODEL_PATH else None
)

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

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    dataloaders=dataloaders,
    dataset=dataset,
    device=device,
    original_model=original_model,
    debug=DEBUG,
    evaluate=EVALUATE_MODEL,
    save=SAVE_MODEL,
)

trainer.train(EPOCHS)
