import math

import torch
import torch.nn as nn

from src.models.transformer.block.encoder import Encoder
from src.models.transformer.block.decoder import Decoder


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
        # (B, src_len) -> (B, src_len, d_model)
        self.src_emb = src_emb
        # (B, tgt_len) -> )B, tgt_len, d_model)
        self.tgt_emb = tgt_emb

        self.encoder = Encoder(
            src_emb, d_model, n_head, d_hidden, n_layers, drop_prob, device
        )
        self.decoder = Decoder(
            tgt_emb,
            d_model,
            tgt_vocab_size,
            n_head,
            d_hidden,
            n_layers,
            drop_prob,
            device,
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
        # (B, max_len, d_model)
        encoder_output = self.encoder(src, src_mask)

        return self.decoder(
            tgt, encoder_output, tgt_mask, src_mask
        )  # (B, tgt_len, vocab_size)
