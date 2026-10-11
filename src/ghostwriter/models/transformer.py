import sys
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class LyricsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=64,
        max_seq_len=128,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2, norm=nn.LayerNorm(embedding_size)
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, x, return_hidden=False):
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
        # Use math attention for Windows/ROCm compatibility with training dropout.
        attention_backend = (
            sdpa_kernel(SDPBackend.MATH)
            if x.is_cuda and torch.version.hip is not None and sys.platform == "win32"
            else nullcontext()
        )
        with attention_backend:
            hidden = self.transformer(hidden, mask=causal_mask)  # (B, T, E)

        if return_hidden:
            return hidden

        return self.output_layer(hidden)
