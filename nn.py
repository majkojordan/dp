import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=100,
        batch_size=1,
        num_layers=1,
        device=torch.device("cpu"),
    ):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden = None
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.device = device
        self.reset_hidden()

        self = self.to(device)
        print("Model initialized")

    def forward(self, input):
        output, self.hidden = self.gru(input, self.hidden)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        # get only last items
        output = output[torch.arange(output.shape[0]), lengths - 1]
        output = F.relu(output)
        output = self.linear(output)
        # output = F.log_softmax(output, dim=1) # don't use if loss_function is CrossEntropyLoss

        return output

    def reset_hidden(self):
        self.hidden = torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size
        ).to(self.device)
