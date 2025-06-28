from json import decoder
import torch

from src.dataset.transformer_ds import causal_mask

def beam_decode(
    model, beam_size, source, source_mask, tokenizer_tgt, max_len, device
):
  model.eval()
  sos_idx = tokenizer_tgt.token_to_id("[SOS]")
  eos_idx = tokenizer_tgt.token_to_id("[EOS]")
  
  assert source.size(0) == 1, "Batch size must be 1 for beam search"
  # (1, dynamic_seq_len)
  encoder_output = model.encoder(source, source_mask)
  # (1, dynamic_seq_len)
  decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
  
  candidates = [(decoder_input, 0.0)]
  
  while True:
    if any([candidate[0].size(1) == max_len for candidate in candidates]):
      break
    
    new_candidates = []
    
    for candidate, score in candidates:
      
      if candidate[0][-1] == eos_idx:
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
      topk_probs, topk_ids = torch.topk(probs, beam_size,dim=1)
      for k in range(beam_size):
        token_id = topk_ids[0][k].unsqueeze(0).unsqueeze(0)
        prob = topk_probs[0][k]
        # (1, ++dynamic_seq_len)
        new_candidate = torch.cat([candidate, token_id],dim=1)
        new_candidates.append((new_candidate, score + torch.log(prob).item()))
        
    candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_size]
    if all([candidate[0][0][-1].item() == eos_idx for candidate in candidates]):
      break
      
  return candidates[0][0].squeeze()# (dynamic_seq_len)        
