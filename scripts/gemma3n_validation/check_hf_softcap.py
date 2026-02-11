#!/usr/bin/env python3
"""Check if HF uses softcap or any other special attention features."""

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")
dtype = torch.float32

# Model should already be cached
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

layer0 = hf_model.model.language_model.layers[0]

print("="*80)
print("CHECK HF ATTENTION SPECIAL FEATURES")
print("="*80)

print(f"\n[Layer 0 self_attn config]")
cfg = layer0.self_attn.config

# Check for softcap
attrs_to_check = [
    'attn_logit_softcapping',
    'query_pre_attn_scalar',
    'final_logit_softcapping',
    'sliding_window',
    'attention_dropout',
    'num_attention_heads',
    'num_key_value_heads',
]

for attr in attrs_to_check:
    if hasattr(cfg, attr):
        val = getattr(cfg, attr)
        print(f"  {attr}: {val}")

# Check layer 0 self_attn direct attributes
print(f"\n[Layer 0 self_attn attributes]")
direct_attrs = ['sliding_window', 'attention_dropout', 'attn_logit_softcapping']
for attr in direct_attrs:
    if hasattr(layer0.self_attn, attr):
        val = getattr(layer0.self_attn, attr)
        print(f"  {attr}: {val}")

print("="*80)
