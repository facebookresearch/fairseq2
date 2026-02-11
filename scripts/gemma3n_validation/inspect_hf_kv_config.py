#!/usr/bin/env python3
"""Inspect HF KV sharing attributes for all layers."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

hf_lm = hf_model.model.language_model

print("\n" + "="*80)
print("HF KV SHARING CONFIGURATION")
print("="*80)

print(f"\n{'Layer':<8} {'Type':<10} {'is_kv_shared':<15} {'kv_shared_idx':<15} {'store_full_len':<15}")
print("-"*80)

for layer_idx in range(30):
    layer = hf_lm.layers[layer_idx]
    layer_type = layer.attention_type

    is_kv_shared = layer.self_attn.is_kv_shared_layer
    kv_shared_idx = layer.self_attn.kv_shared_layer_index
    store_full_len = layer.self_attn.store_full_length_kv

    # Highlight source and consumer layers
    marker = ""
    if store_full_len:
        marker = "📦 SOURCE"
    elif is_kv_shared:
        marker = f"🔗 CONSUMER (from layer {kv_shared_idx})"

    print(f"{layer_idx:<8} {layer_type:<10} {str(is_kv_shared):<15} {str(kv_shared_idx):<15} {str(store_full_len):<15} {marker}")

print("-"*80)

# Now let's check if there's a global storage mechanism
print("\n\nChecking for global KV storage...")
if hasattr(hf_lm, 'kv_proj_cache'):
    print(f"  hf_lm has kv_proj_cache: {hf_lm.kv_proj_cache}")
else:
    print("  hf_lm does NOT have kv_proj_cache")

# Check if cache is in config
print(f"\nModel config:")
print(f"  num_kv_shared_layers: {hf_lm.config.num_kv_shared_layers}")
print(f"  kv_sinks (if present): {getattr(hf_lm.config, 'kv_sinks', 'N/A')}")
