#!/usr/bin/env python3
"""Test if manual GQA expansion matches PyTorch SDPA's internal broadcasting."""

import torch
from fairseq2.ops import repeat_interleave

device = torch.device("cpu")
dtype = torch.float32

# Create test K/V with 2 KV heads
batch, seq_len, num_kv_heads, head_dim = 1, 4, 2, 256
num_q_heads = 8

torch.manual_seed(42)
k_kv = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
v_kv = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
q = torch.randn(batch, seq_len, num_q_heads, head_dim, dtype=dtype, device=device)

print("="*80)
print("COMPARE GQA METHODS")
print("="*80)

print(f"\nOriginal shapes:")
print(f"  Q: {q.shape} (8 heads)")
print(f"  K: {k_kv.shape} (2 KV heads)")
print(f"  V: {v_kv.shape} (2 KV heads)")

# Method 1: FS2 manual expansion using repeat_interleave
print(f"\n[METHOD 1: FS2 repeat_interleave]")
k_fs2 = repeat_interleave(k_kv, dim=-2, repeat=num_q_heads // num_kv_heads)
v_fs2 = repeat_interleave(v_kv, dim=-2, repeat=num_q_heads // num_kv_heads)
print(f"  K expanded: {k_fs2.shape}")
print(f"  V expanded: {v_fs2.shape}")

# Method 2: PyTorch SDPA with GQA (passes unexpanded K/V)
print(f"\n[METHOD 2: PyTorch SDPA internal GQA]")
# PyTorch SDPA expects [batch, heads, seq, dim]
q_torch = q.transpose(1, 2)  # [batch, 8, seq, dim]
k_torch = k_kv.transpose(1, 2)  # [batch, 2, seq, dim]
v_torch = v_kv.transpose(1, 2)  # [batch, 2, seq, dim]

print(f"  Passing Q={q_torch.shape}, K={k_torch.shape}, V={v_torch.shape}")
print(f"  PyTorch SDPA will internally broadcast K/V from 2 to 8 heads")

# Run both methods
mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
mask.triu_(diagonal=1)

# Method 1: Manual expansion + explicit mask
print(f"\n[RUNNING METHOD 1]")
k_fs2_t = k_fs2.transpose(1, 2)
v_fs2_t = v_fs2.transpose(1, 2)
weights1 = torch.matmul(q_torch, k_fs2_t.transpose(-1, -2)) * (head_dim ** -0.5)
weights1 = weights1 + mask
weights1 = torch.nn.functional.softmax(weights1, dim=-1, dtype=torch.float32).to(dtype)
out1 = torch.matmul(weights1, v_fs2_t)
print(f"  Output shape: {out1.shape}")
print(f"  Output[0, 3, 3, :5]: {out1[0, 3, 3, :5]}")

# Method 2: PyTorch SDPA with is_causal
print(f"\n[RUNNING METHOD 2: is_causal=True]")
out2 = torch.nn.functional.scaled_dot_product_attention(
    q_torch, k_torch, v_torch, attn_mask=None, is_causal=True
)
print(f"  Output shape: {out2.shape}")
print(f"  Output[0, 3, 3, :5]: {out2[0, 3, 3, :5]}")

# Method 3: PyTorch SDPA with explicit mask
print(f"\n[RUNNING METHOD 3: explicit mask]")
out3 = torch.nn.functional.scaled_dot_product_attention(
    q_torch, k_torch, v_torch, attn_mask=mask, is_causal=False
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

print(f"\nManual expansion vs SDPA is_causal: max={diff_1_2.max().item():.6e}")
print(f"Manual expansion vs SDPA explicit mask: max={diff_1_3.max().item():.6e}")
print(f"SDPA is_causal vs SDPA explicit mask: max={diff_2_3.max().item():.6e}")

if diff_1_2.max().item() > 1e-5:
    print(f"\n❌ Manual GQA expansion differs from PyTorch internal GQA!")
elif diff_1_3.max().item() > 1e-5:
    print(f"\n❌ Using explicit mask differs from is_causal=True!")
else:
    print(f"\n✅ All methods match!")

print("="*80)
