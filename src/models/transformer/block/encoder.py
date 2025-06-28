import torch
import torch.nn as nn
from src.models.transformer.attention.multi_head_attention import MultiHeadAttention
from src.models.transformer.layers.feed_forward import FeedForward
from src.models.transformer.layers.residual_connection import ResidualConnection
from src.models.transformer.norm.layer_norm import LayerNorm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_hidden: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            d_model, n_head, device, drop_prob=None
        )
        self.ffn = FeedForward(d_model, d_hidden, drop_prob, device)
        self.residual_connections = [
            ResidualConnection(d_model, drop_prob, device) for _ in range(2)
        ]

    def forward(self, x, src_mask=None):
        """
        Args:
            x (torch.Tensor): (B, seq_len, d_model).
            src_mask (torch.Tensor): (B, 1, 1, seq_len) (optional).

        Returns:
            torch.Tensor: Output tensor of shape (B, seq_len, d_model).
        """
        assert x.dim() == 3, "Input tensor must be of shape (B, seq_len, d_model)"
        if src_mask is not None:
            assert (
                src_mask.dim() == 4
                and src_mask.size(0) == x.size(0)
                and x.size(1) == src_mask.size(3)
            ), "src_mask must be of shape (B, 1, 1, seq_len)"

        # (B, seq_len, d_model)
        x = self.residual_connections[0](
            x, lambda x: self.multi_head_attention(x, x, x, src_mask)
        )
        return self.residual_connections[1](x, self.ffn)  # (B, seq_len, d_model)


class Encoder(nn.Module):
    def __init__(
        self,
        src_emb,
        d_model: int,
        n_head: int,
        d_hidden: int,
        n_layers: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.src_emb = src_emb  # (B, max_len) -> (B, max_len, d_model)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, n_head, d_hidden, drop_prob, device)
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(d_model, device)

    def forward(self, x, src_mask=None):
        """
        Args:
            x (torch.Tensor): (B, max_len).
            src_mask (torch.Tensor): (B, 1, 1, max_len) (optional).

        Returns:
            torch.Tensor: Output tensor of shape (B, max_len, d_model).
        """
        assert x.dim() == 2, "Input tensor must be of shape (B, max_len)"

        # (B, max_len) -> (B, max_len, d_model)
        x = self.src_emb(x)
        assert x.dim() == 3, "Output tensor must be of shape (B, max_len, d_model)"

        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # (B, max_len, d_model)
