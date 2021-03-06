import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        pretrained_embeddings=None,
        embedding_size=100,
        hidden_size=100,
        batch_size=1,
        num_layers=1,
        device=torch.device("cpu"),
        input_dropout=0,
        hidden_dropout=0,
    ):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden = None
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embedding = (
            nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings))
            if pretrained_embeddings is not None
            else nn.Embedding(vocab_size, embedding_size)
        )
        self.dropout = nn.Dropout(input_dropout)
        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=hidden_dropout,
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.device = device
        self.reset_hidden()

        self = self.to(device)
        print("Model initialized")

    def forward(self, input, lengths):
        if len(input) == 0:
            return

        output = self.embedding(input.long())
        output = self.dropout(output)
        output = pack_padded_sequence(
            output, lengths, batch_first=True, enforce_sorted=False
        ).to(self.device)
        output, self.hidden = self.gru(output, self.hidden)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        # get only last items
        output = output[torch.arange(output.shape[0]), lengths - 1]
        output = F.relu(output)
        output = self.linear(output)

        return output

    def reset_hidden(self):
        self.hidden = torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size
        ).to(self.device)
