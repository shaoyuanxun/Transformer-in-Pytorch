import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, device: torch.device, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim)).to(device)
        self.bias = nn.Parameter(torch.zeros(dim)).to(device)

    def forward(self, x):
        """
        Applies layer normalization to the input tensor `x`.

        Args:
            x (torch.Tensor): (B, seq_len, dim)
        Returns:
            torch.Tensor:  (B, seq_len, dim).
        """
        assert x.dim() == 3, "Input tensor must be of shape (B, seq_len, dim)"

        mean = x.mean(dim=-1, keepdim=True)  # (B, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (B, seq_len, 1)
        return (
            self.alpha * (x - mean) / (std + self.eps) + self.bias
        )  # (B, seq_len, dim)
