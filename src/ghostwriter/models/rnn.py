from torch import nn


class LyricsRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=64,
        hidden_size=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        output, _ = self.rnn(emb)
        return self.output_layer(output)
