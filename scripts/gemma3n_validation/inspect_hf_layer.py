#!/usr/bin/env python3
"""Inspect HF layer source to understand what it returns."""

import torch
import inspect
from transformers import AutoModelForCausalLM

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device, local_files_only=True
).eval()

hf_layer = hf_model.model.language_model.layers[0]

print("="*80)
print("HF Layer Source Code Inspection")
print("="*80)

# Get the forward method source
print("\nLayer class:", type(hf_layer).__name__)
print("\nLayer forward signature:")
print(inspect.signature(hf_layer.forward))

# Try to get source (might not work in all environments)
try:
    source = inspect.getsource(hf_layer.forward)
    print("\nForward method source (last 100 lines):")
    lines = source.split('\n')
    for line in lines[-100:]:
        print(line)
except:
    print("\nCouldn't retrieve source - checking class hierarchy:")
    for cls in type(hf_layer).__mro__[:5]:
        print(f"  - {cls.__name__}")

# Check what the layer actually contains
print("\n" + "="*80)
print("Layer attributes:")
print("="*80)
for attr in dir(hf_layer):
    if not attr.startswith('_'):
        obj = getattr(hf_layer, attr, None)
        if isinstance(obj, torch.nn.Module):
            print(f"  {attr:30s} : {type(obj).__name__}")

print("\n" + "="*80)
print("Altup-related attributes:")
print("="*80)
if hasattr(hf_layer, 'altup'):
    altup = hf_layer.altup
    print(f"altup type: {type(altup).__name__}")
    for attr in dir(altup):
        if not attr.startswith('_'):
            print(f"  {attr}")

print("\n" + "="*80)
print("Config altup settings:")
print("="*80)
if hasattr(hf_layer, 'config'):
    cfg = hf_layer.config
    for attr in dir(cfg):
        if 'altup' in attr.lower():
            print(f"  {attr}: {getattr(cfg, attr, None)}")
