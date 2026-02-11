#!/usr/bin/env python3
"""Check what SDPA bias is configured vs what's being used."""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model

device = torch.device("cpu")
dtype = torch.float32

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

layer0 = fs2_model.decoder.layers[0]

print("="*80)
print("LAYER 0 SDPA CONFIGURATION")
print("="*80)

print(f"\nLayer 0 SDPA type: {type(layer0.self_attn.sdpa)}")
print(f"Layer 0 SDPA bias: {layer0.self_attn.sdpa.bias}")
print(f"Layer 0 SDPA bias type: {type(layer0.self_attn.sdpa.bias)}")

# Check if it's a sliding window bias
if hasattr(layer0.self_attn.sdpa.bias, 'window_size'):
    print(f"Sliding window size: {layer0.self_attn.sdpa.bias.window_size}")

# Check other layers
print(f"\n{'='*80}")
print("ALL LAYERS SDPA BIAS")
print(f"{'='*80}")

for i, layer in enumerate(fs2_model.decoder.layers):
    bias_type = type(layer.self_attn.sdpa.bias).__name__
    if hasattr(layer.self_attn.sdpa.bias, 'window_size'):
        print(f"Layer {i}: {bias_type} (window={layer.self_attn.sdpa.bias.window_size})")
    else:
        print(f"Layer {i}: {bias_type}")

print("="*80)
