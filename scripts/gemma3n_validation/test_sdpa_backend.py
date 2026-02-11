#!/usr/bin/env python3
"""Test different PyTorch SDPA backends."""

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.backends.cuda import sdp_kernel, SDPBackend

device = torch.device("cpu")
torch.manual_seed(42)

# Create simple test data
B, H, S, D = 1, 8, 2, 256
q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

# Create sliding window mask
seq_len = S
sliding_window = 512
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
    diagonal=1
)
if sliding_window < seq_len:
    window_mask = torch.tril(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
        diagonal=-sliding_window
    )
    causal_mask = causal_mask + window_mask

print("PyTorch SDPA Backend Test")
print("=" * 80)

# Test 1: Default SDPA
out_default = scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
print(f"Default backend: mean={out_default.mean():.6f}, std={out_default.std():.6f}")

# Test 2: Manual computation with float32 upcast (HF style)
scaling = D ** -0.5
attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
attn_weights = attn_weights + causal_mask
attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
out_manual = torch.matmul(attn_weights, v)
print(f"Manual (HF style): mean={out_manual.mean():.6f}, std={out_manual.std():.6f}")

# Compare
diff = (out_default - out_manual).abs()
print(f"\nDifference: max={diff.max():.6e}, mean={diff.mean():.6e}")

if diff.max() < 1e-6:
    print("✓ SDPA matches manual computation exactly")
elif diff.max() < 1e-4:
    print("⚠️  Small numerical difference (acceptable)")
else:
    print("❌ Significant difference - SDPA backend behaves differently")
    print("\nThis explains why FS2 attention differs from HF!")
    print("HF uses manual computation with float32 upcast for softmax.")
    print("PyTorch SDPA might not upcast, causing numerical differences.")
