import torch.nn as nn


class RnnModule(nn.Module):
    RNN_OPTS = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}

    def __init__(
        self, rnn_class, input_size, hidden_size, num_layers=1, bidirectional=False
    ):
        super().__init__()

        rnn_class = RnnModule.RNN_OPTS[rnn_class]
        self.rnn = rnn_class(
            input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
        )

    def forward(self, x, h):
        x, h = self.rnn(x, h)
        return x, h
