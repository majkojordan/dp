import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import timedelta
from nn import RNN

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


dataset = SequenceDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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
for i in range(1000):
    for (batch, labels) in dataloader:
        optimizer.zero_grad()
        model.reset_hidden()

        y_pred = model(batch)

        label_indexes = torch.argmax(labels, axis=1)
        loss = loss_function(y_pred, label_indexes)
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")

# sequence = [one_hot_encode(i) for i in s.iloc[0]]
# input = torch.tensor(sequence[:-1]).view(1, -1, dataset.item_count)
# label = torch.tensor(sequence[-1])

# # s = s.apply(lambda x: [one_hot_encode(i) for i in x])
# # transform array of (input, label) tuples
# inputs, labels = [(torch.tensor(x[:-1]), torch.tensor(x[-1])) for x in s.tolist()]
# print(inputs)

# padded = nn.utils.rnn.pad_sequence(input, batch_first=True)
# print(nn.utils.rnn.pack_padded_sequence(padded, batch_first=True, lengths=[2]))
# epochs = 10
# aggregated_losses = []
#
# for i in range(epochs):
#     y_pred = model()
#     single_loss = loss_function(y_pred, train_outputs)
#     aggregated_losses.append(single_loss)


#     optimizer.zero_grad()
#     single_loss.backward()
#     optimizer.step()

# print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
