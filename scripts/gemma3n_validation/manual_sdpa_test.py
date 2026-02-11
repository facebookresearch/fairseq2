#!/usr/bin/env python3
"""Manually run SDPA with captured Q, K, V to isolate divergence."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.ops import repeat_interleave

device = torch.device("cpu")
dtype = torch.float32

# Load models and create test input (short for debugging)
hf_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E2B-it", torch_dtype=dtype, device_map=device
).eval()

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()
fs2_model.load_state_dict(convert_gemma3n_state_dict(hf_model.state_dict(), config), strict=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E2B-it")
text = "The quick"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

print("="*80)
print("MANUAL SDPA WITH CAPTURED Q, K, V")
print("="*80)

# Use simple synthetic Q, K, V with known values for easier debugging
bsz, seq_len = 1, 3
num_q_heads, num_kv_heads = 8, 2
head_dim = 256

torch.manual_seed(42)
q_hf = torch.randn(bsz, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
k_kv_hf = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
v_kv_hf = torch.randn(bsz, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

# Convert to FS2 format
q_fs2 = q_hf.transpose(1, 2)  # [batch, seq, heads, dim]
k_kv_fs2 = k_kv_hf.transpose(1, 2)  # [batch, seq, kv_heads, dim]
v_kv_fs2 = v_kv_hf.transpose(1, 2)

print(f"\nQ shape: HF={q_hf.shape}, FS2={q_fs2.shape}")
print(f"K shape (before repeat): HF={k_kv_hf.shape}, FS2={k_kv_fs2.shape}")
print(f"V shape (before repeat): HF={v_kv_hf.shape}, FS2={v_kv_fs2.shape}")

# HF: repeat K/V for GQA
def repeat_kv_hf(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

k_hf = repeat_kv_hf(k_kv_hf, num_q_heads // num_kv_heads)
v_hf = repeat_kv_hf(v_kv_hf, num_q_heads // num_kv_heads)

print(f"\nAfter GQA repeat:")
print(f"  HF K: {k_hf.shape}, V: {v_hf.shape}")

# FS2: repeat K/V using repeat_interleave
k_fs2 = repeat_interleave(k_kv_fs2, dim=-2, repeat=num_q_heads // num_kv_heads)
v_fs2 = repeat_interleave(v_kv_fs2, dim=-2, repeat=num_q_heads // num_kv_heads)

print(f"  FS2 K: {k_fs2.shape}, V: {v_fs2.shape}")

# Verify repetition matches
k_fs2_transposed = k_fs2.transpose(1, 2)
v_fs2_transposed = v_fs2.transpose(1, 2)
diff_k = (k_hf - k_fs2_transposed).abs()
diff_v = (v_hf - v_fs2_transposed).abs()
print(f"\nRepetition diff: K max={diff_k.max().item():.6e}, V max={diff_v.max().item():.6e}")

# Create mask
mask = torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
mask.triu_(diagonal=1)

# HF SDPA
print(f"\n[HF SDPA COMPUTATION]")
attn_weights_hf = torch.matmul(q_hf, k_hf.transpose(-1, -2)) * (head_dim ** -0.5)
print(f"  After matmul: shape={attn_weights_hf.shape}, mean={attn_weights_hf.mean().item():.6f}")

attn_weights_hf = attn_weights_hf + mask
print(f"  After mask: mean={attn_weights_hf.mean().item():.6f}")

attn_weights_hf_fp32 = torch.nn.functional.softmax(attn_weights_hf, dim=-1, dtype=torch.float32)
print(f"  After softmax (fp32): mean={attn_weights_hf_fp32.mean().item():.6f}")

attn_weights_hf = attn_weights_hf_fp32.to(dtype)
attn_output_hf = torch.matmul(attn_weights_hf, v_hf)
print(f"  Final output: shape={attn_output_hf.shape}, mean={attn_output_hf.mean().item():.6f}, std={attn_output_hf.std().item():.6f}")

# FS2 SDPA (using NaiveSDPA logic)
print(f"\n[FS2 SDPA COMPUTATION]")
# FS2 transposes before matmul
q_fs2_t = q_fs2.transpose(-2, -3)  # [batch, heads, seq, dim]
k_fs2_t = k_fs2.transpose(-2, -3)
v_fs2_t = v_fs2.transpose(-2, -3)

attn_weights_fs2 = torch.matmul(q_fs2_t, k_fs2_t.transpose(-1, -2)) * (head_dim ** -0.5)
print(f"  After matmul: shape={attn_weights_fs2.shape}, mean={attn_weights_fs2.mean().item():.6f}")

attn_weights_fs2 = attn_weights_fs2 + mask
print(f"  After mask: mean={attn_weights_fs2.mean().item():.6f}")

attn_weights_fs2_fp32 = torch.nn.functional.softmax(attn_weights_fs2, dim=-1, dtype=torch.float32)
print(f"  After softmax (fp32): mean={attn_weights_fs2_fp32.mean().item():.6f}")

attn_weights_fs2 = attn_weights_fs2_fp32.to(dtype)
attn_output_fs2_t = torch.matmul(attn_weights_fs2, v_fs2_t)
print(f"  Final output: shape={attn_output_fs2_t.shape}, mean={attn_output_fs2_t.mean().item():.6f}, std={attn_output_fs2_t.std().item():.6f}")

# Compare
print(f"\n[COMPARISON]")
diff = (attn_output_hf - attn_output_fs2_t).abs()
print(f"Attention output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

# Also compare attention weights
diff_weights = (attn_weights_hf - attn_weights_fs2).abs()
print(f"Attention weights diff: max={diff_weights.max().item():.6e}, mean={diff_weights.mean().item():.6e}")

print("="*80)
