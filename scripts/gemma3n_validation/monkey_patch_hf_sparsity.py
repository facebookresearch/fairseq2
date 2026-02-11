#!/usr/bin/env python3
"""Verify and monkey-patch HF sparsity."""

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")

print("Loading HF model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    torch_dtype=torch.float32,
    device_map=device,
    local_files_only=True
).eval()

print(f"Model type: {type(hf_model)}")
print(f"Config type: {type(hf_model.config)}")

# Check if text config exists
if hasattr(hf_model.config, 'text_config'):
    print(f"\nText config type: {type(hf_model.config.text_config)}")
    if hasattr(hf_model.config.text_config, 'activation_sparsity_pattern'):
        print(f"Original activation_sparsity_pattern: {hf_model.config.text_config.activation_sparsity_pattern}")
    else:
        print("No activation_sparsity_pattern in text_config")
        print(f"Text config attributes: {[a for a in dir(hf_model.config.text_config) if not a.startswith('_')][:20]}")
else:
    print("No text_config found")
    print(f"Config attributes: {[a for a in dir(hf_model.config) if not a.startswith('_')][:20]}")

# Check actual layers
print("\n" + "="*80)
print("Checking layer MLPs directly:")
print("="*80)

lm = hf_model.model.language_model
for i in [0, 9, 29]:
    if hasattr(lm.layers[i], 'mlp') and hasattr(lm.layers[i].mlp, 'activation_sparsity'):
        sparsity = lm.layers[i].mlp.activation_sparsity
        print(f"Layer {i:2d} MLP activation_sparsity: {sparsity}")

print("\n" + "="*80)
print("Monkey-patching all MLPs to 0.0 sparsity...")
print("="*80)

for i, layer in enumerate(lm.layers):
    if hasattr(layer, 'mlp'):
        old_sparsity = layer.mlp.activation_sparsity
        layer.mlp.activation_sparsity = 0.0
        if i < 3 or i >= len(lm.layers) - 3:
            print(f"Layer {i:2d}: {old_sparsity} -> {layer.mlp.activation_sparsity}")

print("\n✓ All MLPs patched to 0.0 sparsity")

# Verify
print("\n" + "="*80)
print("Verification:")
print("="*80)
for i in [0, 9, 29]:
    print(f"Layer {i:2d} MLP activation_sparsity: {lm.layers[i].mlp.activation_sparsity}")
