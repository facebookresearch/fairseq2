#!/usr/bin/env python3
"""Manually run SDPA on both sides with identical inputs."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout
from fairseq2.ops import repeat_interleave

device = torch.device("cpu")
dtype = torch.float32

hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick"  # Short text for easier debugging
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("MANUAL SDPA COMPARISON WITH IDENTICAL INPUTS")
print("="*80)

# Create simple test tensors with known values
bsz, seq_len = 1, input_ids.shape[1]
num_q_heads, num_kv_heads = 8, 2
head_dim = 256

print(f"\nSeq length: {seq_len}")
print(f"Q heads: {num_q_heads}, KV heads: {num_kv_heads}, head_dim: {head_dim}")

# Create identical Q, K, V tensors
torch.manual_seed(42)
q = torch.randn(bsz, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
k_kv = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
v_kv = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

print(f"\nQ shape: {q.shape}")
print(f"K (before repeat) shape: {k_kv.shape}")
print(f"V (before repeat) shape: {v_kv.shape}")

# HF repeats K/V for GQA
def repeat_kv_hf(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

k_hf = repeat_kv_hf(k_kv, num_q_heads // num_kv_heads)
v_hf = repeat_kv_hf(v_kv, num_q_heads // num_kv_heads)

print(f"\nHF K (after repeat) shape: {k_hf.shape}")
print(f"HF V (after repeat) shape: {v_hf.shape}")

# FS2 repeats K/V (need to convert to FS2 format first)
# FS2 uses (N, S, H, K) format, HF uses (N, H, S, K)
k_kv_fs2 = k_kv.transpose(1, 2)  # -> (batch, seq, kv_heads, head_dim)
v_kv_fs2 = v_kv.transpose(1, 2)

k_fs2 = repeat_interleave(k_kv_fs2, dim=-2, repeat=num_q_heads // num_kv_heads)
v_fs2 = repeat_interleave(v_kv_fs2, dim=-2, repeat=num_q_heads // num_kv_heads)

k_fs2_transposed = k_fs2.transpose(1, 2)  # -> (batch, heads, seq, head_dim)
v_fs2_transposed = v_fs2.transpose(1, 2)

print(f"FS2 K (after repeat) shape: {k_fs2_transposed.shape}")
print(f"FS2 V (after repeat) shape: {v_fs2_transposed.shape}")

# Verify K/V match after repetition
diff_k = (k_hf - k_fs2_transposed).abs()
diff_v = (v_hf - v_fs2_transposed).abs()
print(f"\nK repetition diff: max={diff_k.max().item():.6e}")
print(f"V repetition diff: max={diff_v.max().item():.6e}")

# Create attention mask
mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
mask.triu_(diagonal=1)  # Causal mask

print(f"\nMask shape: {mask.shape}")

# HF SDPA
print(f"\n[HF SDPA]")
attn_weights_hf = torch.matmul(q, k_hf.transpose(-1, -2)) * (head_dim ** -0.5)
attn_weights_hf = attn_weights_hf + mask
attn_weights_hf = torch.nn.functional.softmax(attn_weights_hf, dim=-1, dtype=torch.float32).to(dtype)
attn_output_hf = torch.matmul(attn_weights_hf, v_hf)

print(f"Attention output shape: {attn_output_hf.shape}")
print(f"Output mean: {attn_output_hf.mean().item():.6f}, std: {attn_output_hf.std().item():.6f}")

# FS2 SDPA (using naive implementation)
print(f"\n[FS2 SDPA]")
attn_weights_fs2 = torch.matmul(q, k_fs2_transposed.transpose(-1, -2)) * (head_dim ** -0.5)
attn_weights_fs2 = attn_weights_fs2 + mask
attn_weights_fs2 = torch.nn.functional.softmax(attn_weights_fs2, dim=-1, dtype=torch.float32).to(dtype)
attn_output_fs2 = torch.matmul(attn_weights_fs2, v_fs2_transposed)

print(f"Attention output shape: {attn_output_fs2.shape}")
print(f"Output mean: {attn_output_fs2.mean().item():.6f}, std: {attn_output_fs2.std().item():.6f}")

# Compare
print(f"\n[COMPARISON]")
diff = (attn_output_hf - attn_output_fs2).abs()
print(f"Attention output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

print("="*80)
