#!/usr/bin/env python3
"""Test layer 0 in isolation vs in sequence to find divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=torch.float32, device_map=device, local_files_only=True
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
input_ids = tokenizer("Hi", return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("Test 1: Layer 0 via full decoder forward")
print("="*80)

with torch.no_grad():
    # Full decoder forward
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

    # Frontend
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    # Full decoder
    fs2_full_output = fs2_model.decoder(seqs, batch_layout, state_bag=state_bag)

    print(f"Full decoder output: shape={fs2_full_output.shape}")
    print(f"  mean={fs2_full_output.mean():.6f}, std={fs2_full_output.std():.6f}")

print("\n" + "="*80)
print("Test 2: Layer 0 called directly")
print("="*80)

with torch.no_grad():
    # Setup same as full decoder
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)
    state_bag = IncrementalStateBag(max_num_steps=input_ids.size(1))

    # Frontend (populates state_bag with PLE)
    seqs, _ = fs2_model.decoder_frontend(input_ids, batch_layout, state_bag=state_bag)

    # 4D stack (same as decoder does)
    hidden_4d = fs2_model.decoder._stack_altup(seqs)

    # Get PLE for layer 0
    per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)
    layer_ple = per_layer_inputs[:, :, 0, :] if per_layer_inputs is not None else None

    # Layer 0 forward
    attn_bias_cache = AttentionBiasCache()
    fs2_layer = fs2_model.decoder.layers[0]
    fs2_direct_output = fs2_layer(
        hidden_4d, batch_layout, attn_bias_cache,
        per_layer_input=layer_ple, state_bag=state_bag
    )

    print(f"Direct layer 0 output: shape={fs2_direct_output.shape}")
    print(f"  mean={fs2_direct_output.mean():.6f}, std={fs2_direct_output.std():.6f}")

print("\n" + "="*80)
print("Comparison")
print("="*80)

# The full decoder returns 3D after unstacking, so we need to get the 4D from decoder
# Actually, let me check if they match at the layer level
print("These should be 4D outputs from layer 0, but full decoder continues processing.")
print("We need to compare at the same point in the pipeline.")
