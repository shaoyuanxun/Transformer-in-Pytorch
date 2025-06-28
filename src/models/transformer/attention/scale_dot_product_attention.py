import torch
import torch.nn as nn
import math


class ScaleDocProductAttention(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None
        if drop_prob is not None:
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch, n_head, q_len, d_model/n_head]
            k: [batch, n_head, kv_len, d_model/n_head]
            v: [batch, n_head, kv_len, d_model/n_head]
            mask: [batch, n_head, q_len, kv_len]

        Returns:
            v: [batch, n_head, q_len, d_model/n_head]
            score: [batch, n_head, q_len, kv_len]
        """
        assert (
            q.dim() == k.dim() == v.dim() == 4
        ), "input tensors must have 4 dimensions"
        assert k.size() == v.size(), "k and v must have the same shape"
        assert (
            q.size()[0:2] == k.size()[0:2] == v.size()[0:2]
        ), "q, k, and v must have the same batch size and number of heads"
        assert (
            q.size(3) == k.size(3) == v.size(3)
        ), "d_model/n_head must be the same for q, k, and v"

        dim_per_head = q.size()[3]

        # [batch, n_head, q_len, kv_len]
        scores = q @ k.transpose(2, 3) / math.sqrt(dim_per_head)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e6)
        scores = self.softmax(scores)  # [batch, n_head, q_len, kv_len]
        if self.dropout is not None:
            scores = self.dropout(scores)

        values = scores @ v  # [batch, n_head, q_len, d_model/n_head]

        return values, scores
