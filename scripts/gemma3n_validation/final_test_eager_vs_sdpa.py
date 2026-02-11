#!/usr/bin/env python3
"""
Final test: Compare HF eager_attention_forward vs PyTorch scaled_dot_product_attention
with identical Q, K, V to find the divergence cause.
"""

import torch

device = torch.device("cpu")
dtype = torch.float32

print("="*80)
print("FINAL TEST: HF EAGER vs PYTORCH SDPA")
print("="*80)

# HF's repeat_kv (for GQA expansion)
def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Create test data
batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 4, 8, 2, 256

torch.manual_seed(42)
# Generate Q, K, V in HF format [batch, heads, seq, dim]
q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=dtype, device=device)
k_kv = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
v_kv = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)

print(f"\nInput shapes:")
print(f"  Q: {q.shape} (8 Q heads)")
print(f"  K: {k_kv.shape} (2 KV heads)")
print(f"  V: {v_kv.shape} (2 KV heads)")

# Method 1: HF eager_attention_forward logic
print(f"\n{'='*80}")
print("[METHOD 1: HF EAGER PATH]")
print(f"{'='*80}")

scaling = head_dim ** -0.5
k_expanded = repeat_kv(k_kv, num_q_heads // num_kv_heads)
v_expanded = repeat_kv(v_kv, num_q_heads // num_kv_heads)

print(f"After GQA expansion: K={k_expanded.shape}, V={v_expanded.shape}")

# Compute attention
attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * scaling

# Add causal mask
mask = torch.zeros((batch, 1, seq_len, seq_len), dtype=dtype, device=device)
mask = mask.masked_fill(
    torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool(),
    float('-inf')
)
attn_weights = attn_weights + mask

# Softmax in fp32, then downcast
attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

# Apply to V
hf_output = torch.matmul(attn_weights, v_expanded)
hf_output = hf_output.transpose(1, 2).contiguous()  # -> [batch, seq, heads, dim]

print(f"HF output shape: {hf_output.shape}")
print(f"HF output[0, 3, 3, :5]: {hf_output[0, 3, 3, :5]}")

# Method 2: PyTorch scaled_dot_product_attention (what FS2 uses)
print(f"\n{'='*80}")
print("[METHOD 2: PYTORCH SDPA (FS2 PATH)]")
print(f"{'='*80}")

# First expand K/V (FS2 does this before calling SDPA)
k_expanded_fs2 = repeat_kv(k_kv, num_q_heads // num_kv_heads)
v_expanded_fs2 = repeat_kv(v_kv, num_q_heads // num_kv_heads)

print(f"After GQA expansion: K={k_expanded_fs2.shape}, V={v_expanded_fs2.shape}")

# Call PyTorch SDPA with explicit mask (not is_causal, to match FS2's sliding window path)
mask_2d = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
mask_2d.triu_(diagonal=1)

print(f"Calling scaled_dot_product_attention with explicit mask")
fs2_output = torch.nn.functional.scaled_dot_product_attention(
    q, k_expanded_fs2, v_expanded_fs2,
    attn_mask=mask_2d,
    is_causal=False
)
fs2_output = fs2_output.transpose(1, 2).contiguous()  # -> [batch, seq, heads, dim]

print(f"FS2 output shape: {fs2_output.shape}")
print(f"FS2 output[0, 3, 3, :5]: {fs2_output[0, 3, 3, :5]}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

diff = (hf_output - fs2_output).abs()
print(f"\nMax diff: {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")

if diff.max().item() < 1e-6:
    print(f"\n✅ HF EAGER and PYTORCH SDPA produce IDENTICAL results!")
    print(f"   The implementations are equivalent")
    print(f"   Divergence must be from different Q/K/V inputs")
else:
    print(f"\n❌ HF EAGER and PYTORCH SDPA DIVERGE!")
    print(f"   This is the root cause - implementation difference")

    # Show where they differ
    max_idx = diff.argmax()
    max_pos = torch.unravel_index(max_idx, diff.shape)
    print(f"\nMax diff at position {max_pos}:")
    print(f"  HF:  {hf_output[max_pos].item():.6f}")
    print(f"  FS2: {fs2_output[max_pos].item():.6f}")

print("="*80)
