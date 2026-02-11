#!/usr/bin/env python3
"""Verify altup_projections weights are loaded correctly."""

import torch
from transformers import AutoModelForCausalLM
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

# Load converted checkpoint
converted = convert_gemma3n_state_dict(hf_model.state_dict(), config)
fs2_model.load_state_dict(converted, strict=False)

print("="*80)
print("ALTUP PROJECTION WEIGHT COMPARISON")
print("="*80)

# Compare altup_projections (stack operation)
print("\n[ALTUP PROJECTIONS - Stack]")
for i in range(3):
    hf_weight = hf_model.model.language_model.altup_projections[i].weight
    fs2_weight = fs2_model.decoder.altup_projections[i].weight

    diff = (hf_weight - fs2_weight).abs()
    print(f"\nProjection {i}:")
    print(f"  HF shape:  {hf_weight.shape}")
    print(f"  FS2 shape: {fs2_weight.shape}")
    print(f"  Max diff:  {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")
    print(f"  Weights match: {torch.allclose(hf_weight, fs2_weight, atol=1e-6)}")

# Compare altup_unembed_projections (unstack operation)
print("\n[ALTUP UNEMBED PROJECTIONS - Unstack]")
for i in range(3):
    hf_weight = hf_model.model.language_model.altup_unembed_projections[i].weight
    fs2_weight = fs2_model.decoder.altup_unembed_projections[i].weight

    diff = (hf_weight - fs2_weight).abs()
    print(f"\nUnembed Projection {i}:")
    print(f"  HF shape:  {hf_weight.shape}")
    print(f"  FS2 shape: {fs2_weight.shape}")
    print(f"  Max diff:  {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")
    print(f"  Weights match: {torch.allclose(hf_weight, fs2_weight, atol=1e-6)}")

# Test forward pass through projections
print("\n[TESTING ALTUP STACK FORWARD]")
test_input = torch.randn(1, 10, 2048, device=device, dtype=dtype)

target_magnitude = torch.mean(test_input**2, dim=-1, keepdim=True) ** 0.5
epsilon = torch.tensor(1e-5, device=device, dtype=dtype)

for i in range(3):
    # HF projection
    hf_proj = hf_model.model.language_model.altup_projections[i](test_input)
    hf_mag = torch.mean(hf_proj**2, dim=-1, keepdim=True)
    hf_mag = torch.sqrt(torch.maximum(hf_mag, epsilon))
    hf_result = hf_proj * target_magnitude / hf_mag

    # FS2 projection
    fs2_proj = fs2_model.decoder.altup_projections[i](test_input)
    fs2_mag = torch.mean(fs2_proj**2, dim=-1, keepdim=True)
    fs2_mag = torch.sqrt(torch.maximum(fs2_mag, epsilon))
    fs2_result = fs2_proj * target_magnitude / fs2_mag

    diff = (hf_result - fs2_result).abs()
    print(f"\nProjection {i} forward:")
    print(f"  Max diff:  {diff.max().item():.6e}")
    print(f"  Mean diff: {diff.mean().item():.6e}")

print("="*80)
