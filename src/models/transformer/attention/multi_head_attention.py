import torch
import torch.nn as nn
from src.models.transformer.attention.scale_dot_product_attention import (
    ScaleDocProductAttention,
)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, device: torch.device, mask=None, drop_prob=None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_per_head = d_model // n_head
        assert d_model % n_head == 0, "d_model is not divided by n_head"
        self.mask = mask
        self.scores = None

        self.attention = ScaleDocProductAttention(drop_prob)

        self.q_linear = nn.Linear(d_model, d_model).to(device)
        self.k_linear = nn.Linear(d_model, d_model).to(device)
        self.v_linear = nn.Linear(d_model, d_model).to(device)
        self.out_linear = nn.Linear(d_model, d_model).to(device)

    def split(self, tensor):
        """
        Splits the input tensor into multiple heads.

        Args:
            tensor: [batch, seq_len, d_model]

        Returns:
            [batch, n_head, seq_len, d_model/n_head]
        """
        #     [batch, seq_len, d_model]
        #  -> [batch, seq_len, n_head, d_model/n_head]
        #  -> [batch, n_head, seq_len, d_model/n_head]
        return tensor.view(
            tensor.size(0), tensor.size(1), self.n_head, self.dim_per_head
        ).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch, q_len, d_model]
            k: [batch, kv_len, d_model]
            v: [batch, kv_len, d_model]
            mask: defaults to None.

        Returns:
            out: [batch, q_len, d_model]
        """
        assert (
            q.dim() == k.dim() == v.dim() == 3
        ), "input tensors must have 3 dimensions"
        assert (
            q.size(0) == k.size(0) == v.size(0)
        ), "batch size must be the same for q, k, and v"
        assert (
            q.size(2) == k.size(2) == v.size(2)
        ), "d_model must be the same for q, k, and v"

        # [batch, q_len, d_model] -> [batch, q_len, d_model]
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        q = self.split(q)  # [batch, n_head, q_len, d_model/n_head]
        k = self.split(k)  # [batch, n_head, kv_len, d_model/n_head]
        v = self.split(v)  # [batch, n_head, kv_len, d_model/n_head]

        # val: (batch, n_head, q_len, d_model/n_head)
        # self.cores: (batch, n_head, q_len, kv_len)
        val, self.scores = self.attention(q, k, v, self.mask)
        batch = val.size(0)
        q_len = q.size(2)

        # (batch, n_head, q_len, d_model/n_head) -> (batch, q_len, d_model)
        val = val.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)

        return val
