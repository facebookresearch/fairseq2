#!/usr/bin/env python3
"""Check HF's rope_parameters configuration."""

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    torch_dtype=torch.float32,
    device_map=device,
    local_files_only=True,
)

hf_lm = hf_model.model.language_model
config = hf_lm.config

print("="*80)
print("HF RoPE Configuration")
print("="*80)

print(f"\nLayer types: {set(config.layer_types)}")
print(f"\nRoPE parameters:")
for layer_type in set(config.layer_types):
    params = config.rope_parameters.get(layer_type)
    print(f"\n{layer_type}:")
    if params:
        for key, val in params.items():
            print(f"  {key}: {val}")
    else:
        print(f"  None")

print(f"\nGeneral RoPE config:")
print(f"  rope_theta: {config.rope_theta}")
print(f"  rope_theta_global: {config.rope_theta_global}")
print(f"  head_dim: {config.head_dim}")

# Check actual inv_freq for sliding_attention
rope_emb = hf_lm.rotary_emb
if hasattr(rope_emb, 'sliding_attention_inv_freq'):
    inv_freq = rope_emb.sliding_attention_inv_freq
    print(f"\nsliding_attention inv_freq shape: {inv_freq.shape}")
    print(f"sliding_attention inv_freq sample: {inv_freq[:5]}")

    # Reconstruct what theta this corresponds to
    # inv_freq[i] = 1 / (theta ^ (2*i / dim))
    # So theta = (1 / inv_freq[0]) ^ (dim / 0) -- but this is inf for i=0
    # Let's use i=1: inv_freq[1] = 1 / (theta ^ (2/dim))
    # theta = (1 / inv_freq[1]) ^ (dim/2)
    if len(inv_freq) > 1:
        theta_reconstructed = (1.0 / inv_freq[1].item()) ** (config.head_dim / 2.0)
        print(f"Reconstructed theta from inv_freq: {theta_reconstructed:.1f}")

if hasattr(rope_emb, 'global_attention_inv_freq'):
    inv_freq = rope_emb.global_attention_inv_freq
    print(f"\nglobal_attention inv_freq shape: {inv_freq.shape}")
    print(f"global_attention inv_freq sample: {inv_freq[:5]}")

    if len(inv_freq) > 1:
        theta_reconstructed = (1.0 / inv_freq[1].item()) ** (config.head_dim / 2.0)
        print(f"Reconstructed theta from inv_freq: {theta_reconstructed:.1f}")
