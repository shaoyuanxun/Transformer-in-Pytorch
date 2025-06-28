import torch
import torch.nn as nn
import math


class TokenEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model).to(device)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, seq_len)

        Returns:
            torch.Tensor: (B, seq_len, d_model)
        """
        assert x.dim() == 2, "Input tensor must be of shape (B, seq_len)"

        # (B, seq_len) -> (B, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
