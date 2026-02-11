#!/usr/bin/env python3
"""Check GELU approximation used in fairseq2 vs HF."""

import torch
from torch.nn import GELU

# Test input
x = torch.tensor([0.5, 1.0, 1.5, -0.5])

# fairseq2's GELU (default torch.nn.GELU)
fs2_gelu = GELU()
fs2_output = fs2_gelu(x)

# HF's GELU (with tanh approximation)
hf_output = torch.nn.functional.gelu(x, approximate="tanh")

# Exact GELU (no approximation)
exact_output = torch.nn.functional.gelu(x, approximate="none")

print("Input:", x)
print(f"\nFS2 GELU (torch.nn.GELU): {fs2_output}")
print(f"HF GELU (tanh approx):     {hf_output}")
print(f"Exact GELU (no approx):    {exact_output}")

print(f"\nFS2 vs HF diff: {(fs2_output - hf_output).abs().max():.6e}")
print(f"FS2 vs Exact diff: {(fs2_output - exact_output).abs().max():.6e}")

if (fs2_output - exact_output).abs().max() < 1e-7:
    print("\n✓ FS2 uses exact GELU (no approximation)")
elif (fs2_output - hf_output).abs().max() < 1e-7:
    print("\n✓ FS2 uses tanh approximation (matches HF)")
else:
    print("\n❌ FS2 uses unknown GELU variant")

print(f"\nTo fix: replace GELU() with lambda x: torch.nn.functional.gelu(x, approximate='tanh')")
