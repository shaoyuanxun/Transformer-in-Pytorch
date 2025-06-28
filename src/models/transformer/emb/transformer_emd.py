import torch
import torch.nn as nn
from src.models.transformer.emb.token_emb import TokenEmbeddings
from src.models.transformer.emb.positional_emb import PositionalEncoding


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_embeddings = TokenEmbeddings(d_model, vocab_size, device)
        self.positional_encoding = PositionalEncoding(d_model, self.max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, seq_len)
        Returns:
            torch.Tensor: (B, seq_len, d_model)
        """
        assert x.dim() == 2, "Input tensor must be of shape (B, seq_len)"

        seq_len = x.size()[1]
        assert seq_len <= self.max_len, "seq_len exceeds max_len"

        # Add token embeddings and positional encoding
        # (B, seq_len) -> (B, seq_len, d_model)
        return self.dropout(
            self.token_embeddings(x) + self.positional_encoding(seq_len)
        )
