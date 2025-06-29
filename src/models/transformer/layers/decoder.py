import torch
import torch.nn as nn

from src.models.transformer.attention.multi_head_attention import MultiHeadAttention
from src.models.transformer.layers.feed_forward import FeedForward
from src.models.transformer.layers.residual_connection import ResidualConnection
from src.models.transformer.norm.layer_norm import LayerNorm


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_hidden: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, n_head, device, drop_prob=None
        )
        self.cross_attention = MultiHeadAttention(
            d_model, n_head, device, drop_prob=None
        )
        self.ffn = FeedForward(d_model, d_hidden, drop_prob, device)
        self.residual_connections = [
            ResidualConnection(d_model, drop_prob, device) for _ in range(3)
        ]

    def forward(self, x, encoder_ouput, tgt_mask=None, src_mask=None):
        """
        Args:
            x (torch.Tensor): (B, tgt_len, d_model).
            encoder_ouput (torch.Tensor): (B, max_len, d_model).
            tgt_mask (torch.Tensor): (B, 1, tgt_len, tgt_len) (optional).
            src_mask (torch.Tensor): (B, 1, tgt_len, max_len) (optional).

        Returns:
            torch.Tensor: Output tensor of shape (B, tgt_len, d_model).
        """
        # Self-attention, (B, tgt_len, d_model)
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )

        # Cross-attention, (B, tgt_len, d_model)
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, encoder_ouput, encoder_ouput, src_mask)
        )

        # Feed-forward network, (B, tgt_len, d_model)
        return self.residual_connections[2](x, self.ffn)


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_emb,
        d_model: int,
        tgt_vocab_size: int,
        n_head: int,
        d_hidden: int,
        n_layers: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.tgt_emd = tgt_emb  # (B, tgt_len) -> (B, tgt_len, d_model)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, n_head, d_hidden, drop_prob, device)
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(d_model, device)
        self.linear_proj = nn.Linear(d_model, tgt_vocab_size).to(device)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Performs a forward pass through the Decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, tgt_len).
            encoder_output (torch.Tensor): (B, max_len, d_model).
            tgt_mask (torch.Tensor, optional):(B, 1, tgt_len, tgt_len).
            src_mask (torch.Tensor, optional): (B, 1, tgt_len, max_len).

        Returns:
            torch.Tensor: Output tensor of shape (B, tgt_len, tgt_vocab_size).
        """
        # target embedding, (B, tgt_len) -> (B, tgt_len, d_model)
        assert x.dim() == 2, "Input tensor must be of shape (B, tgt_len)"
        x = self.tgt_emd(x)
        assert x.dim() == 3, "Output tensor must be of shape (B, tgt_len, d_model)"

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return self.linear_proj(self.norm(x))  # (B, tgt_len, tgt_vocab_size)
