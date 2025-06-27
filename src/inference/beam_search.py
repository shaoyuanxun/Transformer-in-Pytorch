from json import decoder
import torch

from src.dataset.transformer_ds import causal_mask

def beam_decode(
    model, beam_size, source, source_mask, tokenizer_tgt, max_len, device
):
  sos_idx = tokenizer_tgt.token_to_id("[SOS]")
  eos_idx = tokenizer_tgt.token_to_id("[EOS]")
  
  assert source.size(0) == 1, "Batch size must be 1 for beam search"
  # (1, dynamic_seq_len)
  encoder_output = model.encoder(source, source_mask)
  # (1, dynamic_seq_len)
  decoder_input = torch.empty(beam_size, 1).fill_(sos_idx).type_as(source).to(device)
  
  candidates = [(decoder_input, 1.0)]
  
  while True:
    if any([candidate[0].size(1) == max_len for candidate in candidates]) == max_len:
      break
    
    new_candidates = []
    
    for candidate, score in candidates:
      
      if candidate[0][-1] == eos_idx:
        continue 
      
      decoder_mask = causal_mask(candidate.size(1)).to(device)
      # (1, dynamic_seq_len, vocab_size)
      proj_output = model.decoder(
          candidate, encoder_output, decoder_mask, source_mask
      )
      
      # (1, dynamic_seq_len, vocab_size) -> (1, vocab_size)
      prob = proj_output[:, -1].softmax(dim=-1)
      # (1, beam_size)
      topk_probs, topk_ids = torch.topk(prob, beam_size,dim=-1)
      
      for k in range(beam_size):
        token_id = topk_ids[0][k]
        prob = topk_probs[0][k].item()
        
        new_candidate = torch.cat([candidate, ],dim=)
      
            