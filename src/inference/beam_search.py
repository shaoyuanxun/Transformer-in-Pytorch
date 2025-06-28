from json import decoder
import torch

from src.dataset.transformer_ds import causal_mask


def beam_decode(
    model: torch.nn.Module,
    beam_size: int,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    max_len: int,
    device: torch.device,
    print_results=False,
):
    """
    Perform beam search decoding for a given input sequence.

    Args:
        model: Transformer model to use for decoding.
        beam_size: Number of beams to use in the search.
        source:  (1, source_len).
        source_mask: (1, 1, 1, source_len).
        tokenizer_tgt: Tokenizer for the target language.
        max_len: Maximum length of the generated sequence.
        device: Device to use for computation.

    Returns:
        The decoded sequence of shape (dynamic_seq_len).
    """
    model.eval()
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    assert source.size(0) == 1, "Batch size must be 1 for beam search"
    # (1, max_len, d_model)
    encoder_output = model.encoder(source, source_mask)
    # (1, dynamic_seq_len)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    all_candidates = [(decoder_input, 0.0)]

    while True:
        if any([candidate[0].size(1) == max_len for candidate in all_candidates]):
            break

        new_candidates = []

        for candidate, score in all_candidates:
            if candidate[0][-1].item() == eos_idx:
                new_candidates.append((candidate, score))
                continue

            decoder_mask = causal_mask(candidate.size(1)).to(device)
            # (1, dynamic_seq_len, vocab_size)
            proj_output = model.decoder(
                candidate, encoder_output, decoder_mask, source_mask
            )

            # (1, dynamic_seq_len, vocab_size) -> (1, vocab_size)
            probs = proj_output[:, -1].softmax(dim=-1)
            # (1, beam_size)
            topk_probs, topk_ids = torch.topk(probs, beam_size, dim=1)
            for k in range(beam_size):
                token_id = topk_ids[0][k].unsqueeze(0).unsqueeze(0)
                prob = topk_probs[0][k]
                # (1, ++dynamic_seq_len)
                new_candidate = torch.cat([candidate, token_id], dim=1)
                new_candidates.append((new_candidate, score + torch.log(prob).item()))

        all_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[
            :beam_size
        ]
        if all([candidate[0][0][-1].item() == eos_idx for candidate in all_candidates]):
            break
        if print_results:
            for i, (candidate, score) in enumerate(all_candidates):
                print("Beam Search Candidates:")
                print(
                    f"Candidate {i}: {tokenizer_tgt.decode(candidate[0].squeeze().detach().cpu().numpy())}, Score: {score}"
                )
    return all_candidates[0][0].squeeze()  # (dynamic_seq_len)
