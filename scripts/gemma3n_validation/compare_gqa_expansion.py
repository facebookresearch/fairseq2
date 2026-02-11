#!/usr/bin/env python3
"""Compare HF's repeat_kv() with FS2's repeat_interleave() for GQA expansion."""

import torch
from fairseq2.ops import repeat_interleave

device = torch.device("cpu")
dtype = torch.float32

print("="*80)
print("COMPARE GQA EXPANSION METHODS")
print("="*80)

# HF's repeat_kv implementation (from transformers)
def repeat_kv_hf(hidden_states, n_rep):
    """
    HuggingFace implementation of GQA key/value repetition.
    Input: [batch, num_key_value_heads, slen, head_dim]
    Output: [batch, num_key_value_heads * n_rep, slen, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Create test tensor with 2 KV heads
batch, seq_len, num_kv_heads, head_dim = 1, 4, 2, 256
n_rep = 4  # Expand to 8 Q heads

torch.manual_seed(42)
# Create in HF format: [batch, kv_heads, seq, dim]
k_hf_format = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)

print(f"\nInput K tensor (HF format):")
print(f"  Shape: {k_hf_format.shape}")
print(f"  [0, 0, 0, :3]: {k_hf_format[0, 0, 0, :3]}")  # KV head 0, seq 0
print(f"  [0, 1, 0, :3]: {k_hf_format[0, 1, 0, :3]}")  # KV head 1, seq 0

# Method 1: HF repeat_kv
print(f"\n[METHOD 1: HF repeat_kv]")
k_hf_expanded = repeat_kv_hf(k_hf_format, n_rep)
print(f"  Output shape: {k_hf_expanded.shape}")
print(f"  Expected: [batch=1, heads=8, seq=4, dim=256]")

# Show how heads are mapped
print(f"\n  Head mapping (first 3 dims at seq=0):")
for head_idx in range(8):
    kv_head = head_idx // n_rep
    print(f"    Q head {head_idx} uses KV head {kv_head}: {k_hf_expanded[0, head_idx, 0, :3]}")

# Method 2: FS2 repeat_interleave
print(f"\n[METHOD 2: FS2 repeat_interleave]")
# Convert to FS2 format: [batch, seq, kv_heads, dim]
k_fs2_format = k_hf_format.transpose(1, 2)
print(f"  Input (FS2 format): {k_fs2_format.shape}")

k_fs2_expanded = repeat_interleave(k_fs2_format, dim=-2, repeat=n_rep)
print(f"  After repeat_interleave: {k_fs2_expanded.shape}")
print(f"  Expected: [batch=1, seq=4, heads=8, dim=256]")

# Convert back to HF format for comparison
k_fs2_to_hf = k_fs2_expanded.transpose(1, 2)
print(f"  Converted to HF format: {k_fs2_to_hf.shape}")

# Show how heads are mapped
print(f"\n  Head mapping (first 3 dims at seq=0):")
for head_idx in range(8):
    print(f"    Q head {head_idx}: {k_fs2_to_hf[0, head_idx, 0, :3]}")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

diff = (k_hf_expanded - k_fs2_to_hf).abs()
print(f"\nOverall diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

# Check per head
print(f"\nPer-head comparison at seq=0:")
for head_idx in range(8):
    diff_head = (k_hf_expanded[0, head_idx, 0, :] - k_fs2_to_hf[0, head_idx, 0, :]).abs()
    hf_kv_head = head_idx // n_rep

    if diff_head.max().item() > 1e-9:
        print(f"  Head {head_idx} (from KV {hf_kv_head}): DIFFERS by {diff_head.max().item():.6e}")
        print(f"    HF:  {k_hf_expanded[0, head_idx, 0, :5]}")
        print(f"    FS2: {k_fs2_to_hf[0, head_idx, 0, :5]}")
    else:
        print(f"  Head {head_idx} (from KV {hf_kv_head}): ✓ matches")

# Detailed analysis of expansion pattern
print(f"\n{'='*80}")
print("EXPANSION PATTERN ANALYSIS")
print(f"{'='*80}")

print(f"\nHF expand pattern (which KV head each Q head gets):")
hf_pattern = []
for head_idx in range(8):
    # Check which original KV head this came from
    for kv_head in range(2):
        if torch.allclose(k_hf_expanded[0, head_idx, 0, :], k_hf_format[0, kv_head, 0, :]):
            hf_pattern.append(kv_head)
            break
print(f"  {hf_pattern}")

print(f"\nFS2 expand pattern (which KV head each Q head gets):")
fs2_pattern = []
for head_idx in range(8):
    # Check which original KV head this came from
    for kv_head in range(2):
        if torch.allclose(k_fs2_to_hf[0, head_idx, 0, :], k_hf_format[0, kv_head, 0, :]):
            fs2_pattern.append(kv_head)
            break
print(f"  {fs2_pattern}")

if hf_pattern == fs2_pattern:
    print(f"\n✅ EXPANSION PATTERNS MATCH!")
else:
    print(f"\n❌ EXPANSION PATTERNS DIFFER!")
    print(f"   This is the root cause of the divergence!")

print("="*80)
