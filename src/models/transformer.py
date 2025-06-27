import math

import torch
import torch.nn as nn
from pkg_resources import require
from zmq import device


class LayerNorm(nn.Module):
    def __init__(self, dim: int, device: torch.device, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim)).to(device)
        self.bias = nn.Parameter(torch.zeros(dim)).to(device)

    def forward(self, x):
        # x: (B, seq_len, dim)
        mean = x.mean(dim=-1, keepdim=True)  # (B, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (B, seq_len, 1)
        return (
            self.alpha * (x - mean) / (std + self.eps) + self.bias
        )  # (B, seq_len, dim)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float, device: torch.device):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden).to(device)
        self.linear2 = nn.Linear(d_hidden, d_model).to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, seq_len, d_model) -> (B, seq_len, d_hidden) -> (B, seq_len, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TokenEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model).to(device)

    def forward(self, x):
        # (B, seq_len) -> (B, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, d_model, requires_grad=False, device = device)
        # (max_len, 1)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(dim=1)
        # (1, d_model/2)
        div_term = 1 / (10000 ** (torch.arange(0, d_model, step=2).float() / d_model))

        self.pe[:, 0::2] = torch.sin(pos * div_term)  # (max_len, d_model)
        self.pe[:, 1::2] = torch.cos(pos * div_term)  # (max_len, d_model)

    def forward(self, seq_len: int):
        assert seq_len <= self.max_len, "seq_len exceeds max_len"
        return self.pe[:seq_len, :]  # (seq_len, d_model)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, drop_prob:float, device: torch.device):
        super().__init__()
        self.token_embeddings = TokenEmbeddings(d_model, vocab_size, device)
        self.positional_encoding = PositionalEncoding(d_model, max_len,device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # (B, seq_len) -> (B, seq_len, d_model)
        seq_len = x.size()[1]
        return self.dropout(
            self.token_embeddings(x) + self.positional_encoding(seq_len)
        )


class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, drop_prob: float, device: torch.device):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.norm = LayerNorm(d_model, device)

    def forward(self, x, sublayer):
        # (B, seq_len, d_model)
        return x + self.dropout(sublayer(self.norm(x)))


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
            mask: defaults to None.

        Returns:
            v: [batch, n_head, q_len, d_model/n_head]
            score: [batch, n_head, q_len, kv_len]
        """

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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, device: torch.device, mask=None, drop_prob=None):
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


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hidden: int, drop_prob: float, device: torch.device):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, device, drop_prob=drop_prob)
        self.ffn = FeedForward(d_model, d_hidden, drop_prob,device)
        self.residual_connections = [
            ResidualConnection(d_model, drop_prob, device) for _ in range(2)
        ]

    def forward(self, x, src_mask=None):
        # x: (B, seq_len, d_model)
        x = self.residual_connections[0](x, lambda x: self.attention(x, x, x, src_mask))
        return self.residual_connections[1](x, self.ffn)  # (B, seq_len, d_model)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_hidden: int,
        n_layers: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, n_head, d_hidden, drop_prob, device)
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(d_model, device)

    def forward(self, x, src_mask=None):
        # x: (B, max_len, d_model)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)  # (B, max_len, d_model)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_hidden: int, drop_prob: float, device: torch.device):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, device, drop_prob=drop_prob)
        self.cross_attention = MultiHeadAttention(d_model, n_head, device, drop_prob=drop_prob)
        self.ffn = FeedForward(d_model, d_hidden, drop_prob, device)
        self.residual_connections = [
            ResidualConnection(d_model, drop_prob, device) for _ in range(3)
        ]

    def forward(self, x, encoder_ouput, tgt_mask=None, src_mask=None):
        # x: (B, tgt_len, d_model)
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        # encoder_ouput: (B, max_len, d_model)
        # x: (B, tgt_len, d_model)
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, encoder_ouput, encoder_ouput, src_mask)
        )

        return self.residual_connections[2](x, self.ffn)  # (B, tgt_len, d_model)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        tgt_vocab_size: int,
        n_head: int,
        d_hidden: int,
        n_layers: int,
        drop_prob: float,
        device: torch.device
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, n_head, d_hidden, drop_prob, device)
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(d_model, device)
        self.linear_proj = nn.Linear(d_model, tgt_vocab_size).to(device)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # x: (B, tgt_len, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.linear_proj(self.norm(x))  # (B, tgt_len, tgt_vocab_size)


class Transformer(nn.Module):
    def __init__(
        self,
        src_emb,
        tgt_emb,
        tgt_vocab_size: int,
        d_model: int,
        n_head: int,
        d_hidden: int,
        n_layers: int,
        device: torch.device,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        # (B, src_len, d_model)
        self.src_emb = src_emb
        # (B, tgt_len, d_model)
        self.tgt_emb = tgt_emb
        self.encoder = Encoder(d_model, n_head, d_hidden, n_layers, drop_prob, device)
        self.decoder = Decoder(
            d_model, tgt_vocab_size, n_head, d_hidden, n_layers, drop_prob, device
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: [batch, max_len]
            trg: [batch, target_len]
            src_mask: [batch, 1, 1, max_len] (optional)
            tgt_mask: [batch, 1, tgt_len, tgt_len] (optional)

        Returns:
            out: [batch, target_len, dec_voc_size]
        """
        # (B, src_len) -> (B, src_len, d_model)
        src = self.src_emb(src)
        # (B, tgt_len) -> (B, tgt_len, d_model)
        tgt = self.tgt_emb(tgt)
        
        # (B, src_len, d_model)
        encoder_output = self.encoder(src, src_mask)
        
        return self.decoder(
            tgt, encoder_output, tgt_mask, src_mask
        )  # (B, tgt_len, vocab_size)
