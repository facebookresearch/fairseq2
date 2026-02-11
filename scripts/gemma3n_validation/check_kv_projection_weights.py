#!/usr/bin/env python3
"""Check which layers have K/V projection weights in HF checkpoint."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")
model_name = "google/gemma-3n-E2B-it"

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=device, local_files_only=True
)

print("\n" + "="*80)
print("K/V PROJECTION WEIGHT ANALYSIS")
print("="*80)

state_dict = hf_model.state_dict()

print(f"\nChecking which layers have K/V projection weights...")
print(f"\n{'Layer':<8} {'Type':<10} {'Has k_proj':<12} {'Has v_proj':<12} {'Expected Role':<15}")
print("-"*80)

# Expected roles based on our configuration
# Layer 13 = LOCAL SOURCE
# Layer 14 = GLOBAL SOURCE
# Layers 15-29 = CONSUMERS

for layer_idx in range(30):
    layer_type = "LOCAL" if (layer_idx + 1) % 5 != 0 and layer_idx != 29 else "GLOBAL"

    k_proj_key = f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"
    v_proj_key = f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight"

    has_k = k_proj_key in state_dict
    has_v = v_proj_key in state_dict

    # Determine expected role
    if layer_idx < 15:
        if layer_idx == 13 or layer_idx == 14:
            role = "SOURCE"
        else:
            role = "NONE"
    else:
        role = "CONSUMER"

    status = ""
    if role == "CONSUMER" and (has_k or has_v):
        status = "⚠️  Consumer has K/V proj!"
    elif role == "CONSUMER" and not has_k and not has_v:
        status = "✅ Consumer shares K/V"

    print(f"{layer_idx:<8} {layer_type:<10} {str(has_k):<12} {str(has_v):<12} {role:<15} {status}")

print("-"*80)

# Check if any consumer layers reference source layer indices
print("\n\nChecking HF config for KV sharing references...")
config = hf_model.config
text_config = hf_model.model.language_model.config

if hasattr(text_config, 'kv_sinks'):
    print(f"  kv_sinks: {text_config.kv_sinks}")

if hasattr(text_config, 'kv_shared_layer_indices'):
    print(f"  kv_shared_layer_indices: {text_config.kv_shared_layer_indices}")

if hasattr(text_config, 'num_kv_shared_layers'):
    print(f"  num_kv_shared_layers: {text_config.num_kv_shared_layers}")

print("\n" + "="*80)
