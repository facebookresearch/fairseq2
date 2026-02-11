#!/usr/bin/env python3
"""Compare 4D stacking operation between HF and FS2."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("4D STACKING OPERATION COMPARISON")
print("="*80)

with torch.no_grad():
    # Get embeddings
    hf_lm = hf_model.model.language_model
    hf_embeds = hf_lm.embed_tokens(input_ids)

    # Get FS2 embeddings through frontend (applies scaling)
    from fairseq2.nn import BatchLayout
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    from fairseq2.nn import IncrementalStateBag
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))
    fs2_embeds, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    print(f"\n[EMBEDDINGS]")
    print(f"HF shape:  {hf_embeds.shape}")
    print(f"FS2 shape: {fs2_embeds.shape}")
    diff = (hf_embeds - fs2_embeds).abs()
    print(f"Max diff:  {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")

    # HF stacking
    print(f"\n[HF 4D STACKING]")
    hidden_states_0 = hf_embeds
    target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
    epsilon_tensor = torch.tensor(1e-5, device=device, dtype=dtype)

    temp_hidden_states = [hidden_states_0]
    for i in range(1, 4):
        altup_proj = hf_lm.altup_projections[i - 1](hidden_states_0)
        new_magnitude = torch.mean(altup_proj**2, dim=-1, keepdim=True)
        new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor))
        altup_proj = altup_proj * target_magnitude / new_magnitude
        temp_hidden_states.append(altup_proj)

        print(f"  Projection {i-1}: shape={altup_proj.shape}, mean={altup_proj.mean().item():.6f}, std={altup_proj.std().item():.6f}")

    hf_stacked = torch.stack(temp_hidden_states, dim=0)
    print(f"  Final HF stack shape: {hf_stacked.shape}")

    # FS2 stacking
    print(f"\n[FS2 4D STACKING]")
    fs2_stacked = fs2_model.decoder._stack_altup(fs2_embeds)
    print(f"  Final FS2 stack shape: {fs2_stacked.shape}")

    # Compare each prediction
    print(f"\n[COMPARING 4D STACKS]")
    for i in range(4):
        diff = (hf_stacked[i] - fs2_stacked[i]).abs()
        print(f"  Prediction [{i}]: max diff = {diff.max().item():.6e}, mean diff = {diff.mean().item():.6e}")

        # Show sample values
        print(f"    HF  sample: {hf_stacked[i, 0, 0, :5].tolist()}")
        print(f"    FS2 sample: {fs2_stacked[i, 0, 0, :5].tolist()}")

print("="*80)
