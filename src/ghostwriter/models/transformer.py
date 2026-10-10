import torch
from torch import nn


class LyricsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=128,
        max_seq_len=128,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2, norm=nn.LayerNorm(embedding_size)
        )
        for layer in self.transformer.layers:
            if not isinstance(layer, nn.TransformerEncoderLayer):
                continue

            weight = layer.self_attn.in_proj_weight

            if isinstance(weight, torch.Tensor):
                nn.init.xavier_uniform_(weight)

            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        # x: (batch, sequence)
        seq_len = x.size(1)

        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError("Sequence exceeds positional embedding capacity.")

        positions = torch.arange(seq_len, device=x.device)
        token_embeddings = self.embedding(x)  # (B, T, E)
        position_embeddings = self.position_embedding(positions)  # (T, E)
        hidden = self.dropout(token_embeddings + position_embeddings)

        # True means attention to this position is blocked.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)  # (B, T, E)
        return self.output_layer(hidden)
