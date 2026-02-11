#!/usr/bin/env python3
"""Detailed line-by-line trace of TorchSDPA.forward."""

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.transformer.sdpa.torch import TorchSDPA
from fairseq2.nn import BatchLayout
from fairseq2.models.transformer import AttentionBiasCache

device = torch.device("cpu")
dtype = torch.float32

config = get_gemma3n_e2b_config()
fs2_model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

layer0 = fs2_model.decoder.layers[0]
old_sdpa = layer0.self_attn.sdpa

# Manually create TorchSDPA and patch it with detailed logging
class InstrumentedTorchSDPA(TorchSDPA):
    def forward(self, q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
        print(f"\n[InstrumentedTorchSDPA.forward START]")
        print(f"  Step 1: Check needs_weights")

        if needs_weights:
            from fairseq2.error import NotSupportedError
            raise NotSupportedError(f"`{TorchSDPA}` does not support `needs_weights`.")

        print(f"  Step 2: Initialize is_causal = False")
        is_causal = False

        print(f"  Step 3: Check if bias is CausalAttentionBias")
        from fairseq2.models.transformer.attention_bias import CausalAttentionBias
        if isinstance(self.bias, CausalAttentionBias):
            print(f"    Yes, it's CausalAttentionBias")
            print(f"    attn_window_len = {self.bias.attn_window_len}")

            if self.bias.attn_window_len is None:
                print(f"    attn_window_len is None, checking for is_causal optimization")
                full_q = not q_layout.packed and not q_layout.padded
                full_k = not k_layout.packed and not k_layout.padded
                print(f"      full_q={full_q}, full_k={full_k}")

                if full_q and full_k:
                    q_len = q.size(1)
                    k_len = k.size(1)
                    is_causal = q_len == k_len
                    print(f"      q_len={q_len}, k_len={k_len}, is_causal={is_causal}")
            else:
                print(f"    attn_window_len is not None, will use explicit bias")

        print(f"  Step 4: Get bias tensor")
        if is_causal:
            print(f"    is_causal=True, setting bias=None")
            bias = None
        else:
            print(f"    is_causal=False, getting bias tensor from cache")
            from fairseq2.models.transformer.attention_bias import maybe_get_attention_bias_tensor
            bias = maybe_get_attention_bias_tensor(self.bias, q, q_layout, k_layout, bias_cache)
            print(f"    bias shape: {bias.shape if bias is not None else None}")

        print(f"  Step 5: Get dropout_p")
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p
        print(f"    dropout_p = {dropout_p}")

        print(f"  Step 6: Transpose Q, K, V")
        print(f"    Before: Q={q.shape}, K={k.shape}, V={v.shape}")
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        print(f"    After: Q={q.shape}, K={k.shape}, V={v.shape}")

        print(f"  Step 7: Call scaled_dot_product_attention")
        print(f"    Parameters: is_causal={is_causal}, dropout_p={dropout_p}, bias={'None' if bias is None else bias.shape}")

        from torch.nn.functional import scaled_dot_product_attention
        print(f"    Function: {scaled_dot_product_attention}")

        attns = scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=dropout_p, is_causal=is_causal
        )

        print(f"  Step 8: Transpose output")
        print(f"    Before: {attns.shape}")
        attns = attns.transpose(-2, -3)
        print(f"    After: {attns.shape}")

        print(f"[InstrumentedTorchSDPA.forward END]\n")

        return attns, None

new_sdpa = InstrumentedTorchSDPA(old_sdpa.bias, dropout_p=old_sdpa.dropout_p)

print("="*80)
print("TESTING WITH INSTRUMENTED TorchSDPA")
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

# Hook scaled_dot_product_attention
sdpa_called = [False]

original_sdpa = torch.nn.functional.scaled_dot_product_attention

def sdpa_hook(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    sdpa_called[0] = True
    print(f"\n>>> HOOK: scaled_dot_product_attention WAS CALLED! <<<\n")
    return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

torch.nn.functional.scaled_dot_product_attention = sdpa_hook

with torch.no_grad():
    output, weights = new_sdpa(q, layout, k, layout, v, bias_cache, needs_weights=False)

torch.nn.functional.scaled_dot_product_attention = original_sdpa

print(f"\n{'='*80}")
print(f"RESULT: scaled_dot_product_attention was {'CALLED' if sdpa_called[0] else 'NOT CALLED'}")
print(f"{'='*80}")
