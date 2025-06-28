import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()
        self.max_len = max_len
        # (max_len, d_model)
        self.pe = torch.zeros(self.max_len, d_model, requires_grad=False, device=device)
        # (max_len, 1)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(dim=1)
        # (1, d_model/2)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, step=2).float() / d_model))

        self.pe[:, 0::2] = torch.sin(pos * div_term)  # (max_len, d_model)
        self.pe[:, 1::2] = torch.cos(pos * div_term)  # (max_len, d_model)

    def forward(self, seq_len: int):
        assert seq_len <= self.max_len, "seq_len exceeds max_len"
        return self.pe[:seq_len, :]  # (seq_len, d_model)
