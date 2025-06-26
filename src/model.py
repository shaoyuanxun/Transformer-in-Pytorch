import math

import torch
import torch.nn as nn
from pkg_resources import require


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (B, seq_len, dim)
        mean = x.mean(dim=-1, keepdim=True)  # (B, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (B, seq_len, 1)
        return (
            self.alpha * (x - mean) / (std + self.eps) + self.bias
        )  # (B, seq_len, dim)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, seq_len, d_model) -> (B, seq_len, d_hidden) -> (B, seq_len, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TokenEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (B, seq_len) -> (B, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.MOdule):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.max_len = max_len
        self.pe = torch.zeros(d_model, self.max_len, requires_grad=False)
        # (max_len, 1)
        pos = torch.arrange(0, self.max_len, dtype=torch.float).unsqueeze(dim=1)
        # (1, d_model/2)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, step=2).float() / d_model))

        self.pe[:, 0::2] = torch.sin(pos * div_term)  # (max_len, d_model)
        self.pe[:, 1::2] = torch.cos(pos * div_term)  # (max_len, d_model)

        self.register_buffer("pe", self.pe)

    def forward(self, seq_len: int):
        assert seq_len <= self.max_len, "seq_len exceeds max_len"
        return self.pe[:seq_len, :]  # (seq_len, d_model)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, drop_prob):
        super().__init__()
        self.token_embeddings = TokenEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # (B, seq_len) -> (B, seq_len, d_model)
        seq_len = x.size()[1]
        return self.dropout(
            self.token_embeddings(x) + self.positional_encoding(seq_len)
        )


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, drop_prob: float):
        super().__init__()
        self.dropout = nn.Dropput(drop_prob)
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))
