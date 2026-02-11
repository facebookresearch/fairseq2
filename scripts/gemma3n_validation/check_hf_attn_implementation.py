#!/usr/bin/env python3
"""Check which attention implementation HF is actually using."""

import torch
from transformers import AutoModelForCausalLM

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

print("="*80)
print("CHECK HF ATTENTION IMPLEMENTATION")
print("="*80)

layer0 = hf_model.model.language_model.layers[0]

print(f"\nLayer 0 self_attn class: {type(layer0.self_attn).__name__}")
print(f"Layer 0 self_attn module: {layer0.self_attn.__class__.__module__}")

# Check what forward method it's using
import inspect
forward_source = inspect.getsource(layer0.self_attn.forward)

print(f"\n[Checking forward() implementation]")

if 'scaled_dot_product_attention' in forward_source:
    print("✓ Uses scaled_dot_product_attention (SDPA)")
elif 'eager_attention_forward' in forward_source:
    print("✓ Uses eager_attention_forward (manual computation)")
elif 'flash_attn' in forward_source:
    print("✓ Uses Flash Attention")
else:
    print("? Unknown attention implementation")

print(f"\n[Forward method source (first 50 lines)]:")
lines = forward_source.split('\n')[:50]
for i, line in enumerate(lines, 1):
    print(f"{i:3}: {line}")

# Check config
print(f"\n[Attention config]")
if hasattr(layer0.self_attn, 'config'):
    cfg = layer0.self_attn.config
    if hasattr(cfg, '_attn_implementation'):
        print(f"_attn_implementation: {cfg._attn_implementation}")
    if hasattr(cfg, 'attn_implementation'):
        print(f"attn_implementation: {cfg.attn_implementation}")

# Check model config
print(f"\n[Model config]")
if hasattr(hf_model.config, '_attn_implementation'):
    print(f"Model _attn_implementation: {hf_model.config._attn_implementation}")
if hasattr(hf_model.config, 'attn_implementation'):
    print(f"Model attn_implementation: {hf_model.config.attn_implementation}")

print("="*80)
