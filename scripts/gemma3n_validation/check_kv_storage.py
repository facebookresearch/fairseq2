#!/usr/bin/env python3
"""Check if HF stores K/V in a separate attribute instead of cache."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

device = torch.device("cpu")
dtype = torch.float32
model_name = "google/gemma-3n-E2B-it"

print("Loading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

hf_lm = hf_model.model.language_model

print("\n" + "="*80)
print("Checking for KV storage attributes in layers")
print("="*80)

# Check layer 13 (should be source for local)
print("\nLayer 13 (LOCAL SOURCE):")
layer13 = hf_lm.layers[13]
print(f"  Attributes: {[a for a in dir(layer13) if not a.startswith('_') and 'kv' in a.lower()]}")
print(f"  self_attn attributes: {[a for a in dir(layer13.self_attn) if not a.startswith('_') and 'kv' in a.lower()]}")

# Check if there's a kv_proj_cache or similar
if hasattr(layer13.self_attn, 'kv_proj_cache'):
    print(f"  Has kv_proj_cache: {layer13.self_attn.kv_proj_cache}")

if hasattr(layer13.self_attn, 'cached_key'):
    print(f"  Has cached_key")

if hasattr(layer13.self_attn, 'cached_value'):
    print(f"  Has cached_value")

# Check layer config
print(f"\n  Layer config attributes: {[a for a in dir(layer13) if 'kv' in a.lower() or 'source' in a.lower() or 'shared' in a.lower()]}")

# Check the model-level config
print("\n\nModel config:")
text_config = hf_lm.config
print(f"  num_kv_shared_layers: {text_config.num_kv_shared_layers}")

# Check if model has a kv_proj_cache attribute
print(f"\nModel-level attributes with 'kv':")
print(f"  {[a for a in dir(hf_lm) if not a.startswith('_') and 'kv' in a.lower()]}")

if hasattr(hf_lm, 'kv_proj_cache'):
    print(f"\n  hf_lm.kv_proj_cache type: {type(hf_lm.kv_proj_cache)}")
    print(f"  hf_lm.kv_proj_cache: {hf_lm.kv_proj_cache}")
