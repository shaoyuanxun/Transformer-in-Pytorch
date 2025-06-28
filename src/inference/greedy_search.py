import torch

from src.dataset.transformer_ds import causal_mask


def greedy_decode(
    model: torch.nn.Module,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    max_len: int,
    device: torch.device,
    print_result=False,
):
    """
    Perform greedy decoding for a given input sequence.

    Args:
        model: Transformer model to use for decoding.
        source:  (1, source_len).
        source_mask:  (1, 1, 1, source_len).
        tokenizer_tgt: Tokenizer for the target language.
        max_len: Maximum length of the generated sequence.
        device: Device to use for computation.

    Returns:
        The decoded sequence of shape (dynamic_seq_len).
    """
    model.eval()
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encoder(source, source_mask)
    # (1, dynamic_seq_len)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # (1, max_len, d_model)
        encoder_output = model.encoder(source, source_mask)
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        # (1, dynamic_seq_len, vocab_size)
        proj_output = model.decoder(
            decoder_input, encoder_output, decoder_mask, source_mask
        )

        # (1, dynamic_seq_len, vocab_size) -> (1, vocab_size)
        _, next_word_id = torch.max(proj_output[:, -1], dim=1)
        # (1, ++dynamic_seq_len)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word_id.item()).to(device),
            ],
            dim=1,
        )

        if next_word_id == eos_idx:
            break

    if print_result:
        print(
            f"Greedy Search Result: {tokenizer_tgt.decode(decoder_input.squeeze(0).detach().cpu().numpy())}"
        )

    return decoder_input.squeeze(0)  # (dynamic_seq_len)
