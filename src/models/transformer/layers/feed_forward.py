import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int, dropout: float, device: torch.device
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden).to(device)
        self.linear2 = nn.Linear(d_hidden, d_model).to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor of shape (B, seq_len, d_model)
        """
        assert x.dim() == 3, "Input tensor must be of shape (B, seq_len, d_model)"
        # (B, seq_len, d_model) -> (B, seq_len, d_hidden) -> (B, seq_len, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
