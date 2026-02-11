#!/usr/bin/env python3
"""Check if TorchSDPA is raising an exception."""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.transformer.sdpa.torch import TorchSDPA
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

# Switch layer 0 to TorchSDPA
layer0 = fs2_model.decoder.layers[0]
old_sdpa = layer0.self_attn.sdpa
new_sdpa = TorchSDPA(old_sdpa.bias, dropout_p=old_sdpa.dropout_p)
layer0.self_attn.sdpa = new_sdpa

print("="*80)
print("TEST TorchSDPA DIRECTLY")
print("="*80)

# Create test tensors
batch_size = 1
seq_len = 4
num_heads = 8
head_dim = 256

q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

seq_lens = [seq_len]
layout = BatchLayout((batch_size, seq_len), seq_lens, device=device)
bias_cache = AttentionBiasCache()

print(f"\nInput shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
print(f"Layout: packed={layout.packed}, padded={layout.padded}")

# Hook scaled_dot_product_attention
call_log = []

original_sdpa = torch.nn.functional.scaled_dot_product_attention

def sdpa_hook(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    call_log.append({
        'is_causal': is_causal,
        'has_mask': attn_mask is not None,
        'mask_shape': attn_mask.shape if attn_mask is not None else None
    })
    print(f"\n✓ scaled_dot_product_attention called!")
    print(f"  is_causal={is_causal}, mask={attn_mask.shape if attn_mask is not None else None}")
    return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

torch.nn.functional.scaled_dot_product_attention = sdpa_hook

print(f"\n[CALLING TorchSDPA.forward]")
try:
    with torch.no_grad():
        output, weights = new_sdpa(q, layout, k, layout, v, bias_cache, needs_weights=False)

    print(f"\n✓ TorchSDPA.forward succeeded")
    print(f"  Output shape: {output.shape}")
    print(f"  scaled_dot_product_attention was called {len(call_log)} times")

    if call_log:
        print(f"  Call details: {call_log[0]}")
    else:
        print(f"  ⚠️  WARNING: scaled_dot_product_attention was NEVER called!")
        print(f"  This means TorchSDPA is not using PyTorch SDPA")

except Exception as e:
    print(f"\n❌ TorchSDPA.forward raised exception:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

torch.nn.functional.scaled_dot_product_attention = original_sdpa

print("\n" + "="*80)
