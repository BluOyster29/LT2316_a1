from torch import nn

class GRUNetwork(nn.Module):
    def __init__(self, vocab_size, sequence_len, input_size, hidden_size, nr_layers, output_size, device, dropout=0.1):
        super().__init__()
        self.num_layers = nr_layers
        self.seq_len = sequence_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.emb = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size,
                          nr_layers=self.nr_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * seq_lenuence, output_size)

    def forward(self, sequence, hidden_layer):
        output = self.emb(sequence)
        hidden_layer = hidden_layer.to(self.dev)
        output, hidden_layer = self.gru(output, hidden_layer)
        output = output.contiguous().view(-1, self.hidden_size *
                                          len(sequence[0]))
        output = self.fc(output)
        return output, hidden_layer

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).float()
