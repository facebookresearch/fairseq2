#!/usr/bin/env python3
"""Test if is_causal=True matches explicit causal mask in PyTorch SDPA."""

import torch

device = torch.device("cpu")
dtype = torch.float32

# Create test tensors
batch, num_heads, seq_len, head_dim = 1, 8, 4, 256

torch.manual_seed(42)
q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)

print("="*80)
print("COMPARE is_causal=True vs EXPLICIT MASK")
print("="*80)

# Method 1: is_causal=True
print(f"\n[METHOD 1: is_causal=True]")
out1 = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None, is_causal=True
)
print(f"  Output shape: {out1.shape}")
print(f"  Output[0, 3, 3, :5]: {out1[0, 3, 3, :5]}")

# Method 2: Explicit causal mask (what FS2 uses)
print(f"\n[METHOD 2: Explicit causal mask]")
mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
mask.triu_(diagonal=1)
print(f"  Mask:\n{mask}")

out2 = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=mask, is_causal=False
)
print(f"  Output shape: {out2.shape}")
print(f"  Output[0, 3, 3, :5]: {out2[0, 3, 3, :5]}")

# Method 3: Sliding window mask with window=512 (larger than seq_len)
print(f"\n[METHOD 3: Sliding window mask (window=512, seq=4)]")
# For sliding window, mask positions beyond window
window_size = 512
sw_mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
for i in range(seq_len):
    for j in range(seq_len):
        if j > i:  # Future positions
            sw_mask[i, j] = float('-inf')
        elif i - j >= window_size:  # Beyond window
            sw_mask[i, j] = float('-inf')
        else:
            sw_mask[i, j] = 0.0

print(f"  Sliding window mask:\n{sw_mask}")

out3 = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=sw_mask, is_causal=False
)
print(f"  Output shape: {out3.shape}")
print(f"  Output[0, 3, 3, :5]: {out3[0, 3, 3, :5]}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

diff_1_2 = (out1 - out2).abs()
diff_1_3 = (out1 - out3).abs()
diff_2_3 = (out2 - out3).abs()

print(f"\nis_causal vs explicit causal: max={diff_1_2.max().item():.6e}, mean={diff_1_2.mean().item():.6e}")
print(f"is_causal vs sliding window: max={diff_1_3.max().item():.6e}, mean={diff_1_3.mean().item():.6e}")
print(f"explicit causal vs sliding window: max={diff_2_3.max().item():.6e}, mean={diff_2_3.mean().item():.6e}")

if diff_1_2.max().item() > 1e-6:
    print(f"\n❌ is_causal=True differs from explicit mask!")
    print(f"   This could be the divergence cause")
else:
    print(f"\n✅ All methods produce identical results")
    print(f"   The mask type is not the issue")

print("="*80)
