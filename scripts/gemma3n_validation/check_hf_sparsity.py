#!/usr/bin/env python3
"""Verify HF sparsity was actually disabled."""

import torch
from transformers import AutoModelForCausalLM, AutoConfig

device = torch.device("cpu")

# Load with modified config
hf_config = AutoConfig.from_pretrained("google/gemma-3n-E2B-it", local_files_only=True)
print(f"Original activation_sparsity_pattern: {hf_config.activation_sparsity_pattern}")

hf_config.activation_sparsity_pattern = [0.0] * len(hf_config.activation_sparsity_pattern)
print(f"Modified activation_sparsity_pattern: {hf_config.activation_sparsity_pattern}")

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it",
    config=hf_config,
    torch_dtype=torch.float32,
    device_map=device,
    local_files_only=True
).eval()

# Check if it actually took effect
print(f"\nModel config activation_sparsity_pattern: {hf_model.config.activation_sparsity_pattern}")

# Check a specific layer's MLP
layer0_mlp = hf_model.model.language_model.layers[0].mlp
print(f"Layer 0 MLP activation_sparsity: {layer0_mlp.activation_sparsity}")

layer9_mlp = hf_model.model.language_model.layers[9].mlp
print(f"Layer 9 MLP activation_sparsity: {layer9_mlp.activation_sparsity}")

if layer0_mlp.activation_sparsity == 0.0 and layer9_mlp.activation_sparsity == 0.0:
    print("\n✓ HF sparsity successfully disabled")
else:
    print(f"\n❌ HF sparsity NOT disabled! Need to monkey-patch MLPs")
    print("\nMonkey-patching all MLPs to disable sparsity...")
    for layer in hf_model.model.language_model.layers:
        layer.mlp.activation_sparsity = 0.0

    print(f"After patch - Layer 0 MLP: {hf_model.model.language_model.layers[0].mlp.activation_sparsity}")
    print(f"After patch - Layer 9 MLP: {hf_model.model.language_model.layers[9].mlp.activation_sparsity}")
    print("✓ Monkey-patch applied")
