#!/usr/bin/env python3
"""Inspect HF model FFN dimensions per layer."""

import torch
from transformers import AutoModelForCausalLM

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
)

print("Layer FFN dimensions:")
for i, layer in enumerate(hf_model.model.language_model.layers):
    gate_shape = layer.mlp.gate_proj.weight.shape
    print(f"  Layer {i:2d}: gate_proj.weight shape = {gate_shape}, inner_dim = {gate_shape[0]}")

# Also check config
config = hf_model.config
print(f"\nConfig values:")
print(f"  intermediate_size: {config.intermediate_size}")
print(f"  num_hidden_layers: {config.num_hidden_layers}")
if hasattr(config, 'altup_hidden_dim'):
    print(f"  altup_hidden_dim: {config.altup_hidden_dim}")
if hasattr(config, 'intermediate_size_local'):
    print(f"  intermediate_size_local: {config.intermediate_size_local}")
