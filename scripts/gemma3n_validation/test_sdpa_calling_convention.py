#!/usr/bin/env python3
"""Run HF and FS2 SDPA with IDENTICAL captured Q, K, V to isolate calling difference."""

import torch
from transformers import AutoModelForCausalLM
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

print("="*80)
print("RUN BOTH SDPA WITH IDENTICAL INPUTS")
print("="*80)

# Create identical test inputs
bsz, seq_len = 1, 3
num_q_heads, num_kv_heads = 8, 2
head_dim = 256

torch.manual_seed(42)
# Create in HF format first
q_hf = torch.randn(bsz, num_q_heads, seq_len, head_dim, dtype=dtype, device=device)
k_kv_hf = torch.randn(bsz, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
v_kv_hf = torch.randn(bsz, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)

print(f"\nTest inputs: seq_len={seq_len}, q_heads={num_q_heads}, kv_heads={num_kv_heads}")

# Convert to FS2 format
q_fs2 = q_hf.transpose(1, 2)  # [batch, seq, heads, dim]
k_kv_fs2 = k_kv_hf.transpose(1, 2)
v_kv_fs2 = v_kv_hf.transpose(1, 2)

# Get actual SDPA modules
hf_layer0 = hf_model.model.language_model.layers[0]
fs2_layer0 = fs2_model.decoder.layers[0]

# HF SDPA
print(f"\n[HF SDPA]")

# HF repeats K/V internally in eager_attention_forward
def repeat_kv_hf(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

k_hf_repeated = repeat_kv_hf(k_kv_hf, num_q_heads // num_kv_heads)
v_hf_repeated = repeat_kv_hf(v_kv_hf, num_q_heads // num_kv_heads)

# Create mask
mask_hf = torch.zeros((bsz, 1, seq_len, seq_len), dtype=dtype, device=device)
mask_hf = mask_hf.masked_fill(
    torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool(),
    float('-inf')
)

# Run HF SDPA logic manually
attn_weights_hf = torch.matmul(q_hf, k_hf_repeated.transpose(2, 3)) * (head_dim ** -0.5)
attn_weights_hf = attn_weights_hf + mask_hf
attn_weights_hf = torch.nn.functional.softmax(attn_weights_hf, dim=-1, dtype=torch.float32).to(dtype)
hf_output = torch.matmul(attn_weights_hf, v_hf_repeated)
hf_output = hf_output.transpose(1, 2).contiguous()  # -> [batch, seq, heads, dim]

print(f"  Output shape: {hf_output.shape}")
print(f"  Output stats: mean={hf_output.mean().item():.6f}, std={hf_output.std().item():.6f}")

# FS2 SDPA
print(f"\n[FS2 SDPA]")

# FS2 repeats K/V using repeat_interleave
k_fs2_repeated = repeat_interleave(k_kv_fs2, dim=-2, repeat=num_q_heads // num_kv_heads)
v_fs2_repeated = repeat_interleave(v_kv_fs2, dim=-2, repeat=num_q_heads // num_kv_heads)

# Run through actual FS2 SDPA module
from fairseq2.models.transformer import AttentionBiasCache
seq_lens = [seq_len]
batch_layout = BatchLayout((bsz, seq_len), seq_lens, device=device)
bias_cache = AttentionBiasCache()

# Call FS2 SDPA directly
fs2_output, _ = fs2_layer0.self_attn.sdpa(
    q_fs2, batch_layout,
    k_fs2_repeated, batch_layout,
    v_fs2_repeated,
    bias_cache,
    needs_weights=False
)

print(f"  Output shape: {fs2_output.shape}")
print(f"  Output stats: mean={fs2_output.mean().item():.6f}, std={fs2_output.std().item():.6f}")

# Compare
print(f"\n[COMPARISON]")
diff = (hf_output - fs2_output).abs()
print(f"SDPA output diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

if diff.max().item() < 1e-5:
    print("\n✅ SDPA outputs match! Issue is elsewhere.")
else:
    print(f"\n❌ SDPA outputs diverge! Issue is in SDPA calling convention.")
    print(f"\nDEBUG INFO:")
    print(f"  HF output sample [0,0,0,:5]: {hf_output[0,0,0,:5]}")
    print(f"  FS2 output sample [0,0,0,:5]: {fs2_output[0,0,0,:5]}")

print("="*80)
