import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=100, batch_size=1, num_layers=1
    ):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden = None
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation = nn.Softmax(dim=1)

        self.reset_hidden()

    def forward(self, input):
        output, self.hidden = self.gru(input, self.hidden)
        output = output.permute(1, 0, 2)[-1]
        output = self.linear(output)
        output = self.activation(output)

        return output

    def reset_hidden(self):
        self.hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
