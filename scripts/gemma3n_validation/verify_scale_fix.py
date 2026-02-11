#!/usr/bin/env python3
"""
Verify that SDPA has scale=1.0 parameter set.
This confirms the fix is actually being used.
"""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model

device = torch.device("cpu")
dtype = torch.float32

print("="*80)
print("VERIFY SCALING FIX IS APPLIED")
print("="*80)

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

# Check layer 0 SDPA
layer0 = fs2_model.decoder.layers[0]
sdpa = layer0.self_attn.sdpa

print(f"\n[Layer 0 Self-Attention SDPA]")
print(f"  Class: {type(sdpa).__name__}")
print(f"  Module: {type(sdpa).__module__}")

# Check if scale attribute exists
if hasattr(sdpa, 'scale'):
    print(f"  ✅ Has 'scale' attribute: {sdpa.scale}")

    if sdpa.scale == 1.0:
        print(f"  ✅ Scale is correctly set to 1.0")
    elif sdpa.scale is None:
        print(f"  ❌ Scale is None (will use 1/sqrt(d_k) - WRONG!)")
    else:
        print(f"  ⚠️  Scale is {sdpa.scale} (expected 1.0)")
else:
    print(f"  ❌ Does NOT have 'scale' attribute!")
    print(f"  This means the code changes were not applied or not loaded.")
    print(f"  ACTION: Restart Python or force module reload!")

# Show the full module
print(f"\n[SDPA repr]")
print(f"  {sdpa}")

# Check a few more layers
print(f"\n[Other layers]")
for idx in [1, 4, 10, 20]:
    if idx < len(fs2_model.decoder.layers):
        layer_sdpa = fs2_model.decoder.layers[idx].self_attn.sdpa
        has_scale = hasattr(layer_sdpa, 'scale')
        scale_value = layer_sdpa.scale if has_scale else "N/A"
        print(f"  Layer {idx}: scale={scale_value}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if hasattr(sdpa, 'scale') and sdpa.scale == 1.0:
    print("✅ Scaling fix IS applied correctly!")
    print("   If you still see divergence, the issue is elsewhere.")
else:
    print("❌ Scaling fix is NOT applied!")
    print("   SOLUTION: Restart your Python session/kernel and re-run tests.")
    print("   Python caches imported modules - old code may still be loaded.")

print("="*80)
