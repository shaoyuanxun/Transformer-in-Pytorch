import torch
import torch.nn as nn
from src.models.transformer.norm.layer_norm import LayerNorm


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, drop_prob: float, device: torch.device):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.norm = LayerNorm(d_model, device)

    def forward(self, x, sublayer):
        # (B, seq_len, d_model)
        """
        Applies a residual connection followed by layer normalization and dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (B, seq_len, d_model).
            sublayer (Callable): A sublayer function to apply to the normalized input.

        Returns:
            torch.Tensor: Output tensor of shape (B, seq_len, d_model).
        """

        return x + self.dropout(sublayer(self.norm(x)))
