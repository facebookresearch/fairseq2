#!/usr/bin/env python3
"""Debug: Check if layer 20 receives pre_computed_kv."""

from __future__ import annotations

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.kv_projection import KVProjectionType, KVProjectionRole
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

config = get_gemma3n_e2b_config()
model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

# Monkey patch layer 20's self_attn to trace pre_computed_kv
layer20 = model.decoder.layers[20]
print(f"Layer 20 config:")
print(f"  is_global: {layer20.is_global}")
print(f"  kv_projection_role: {layer20.kv_projection_role}")

original_forward = layer20.self_attn.forward

def traced_forward(seqs, seqs_layout, keys, keys_layout, values, bias_cache, *, state_bag=None, pre_computed_kv=None, kv_storage_callback=None):
    print(f"\n  Layer 20 self_attn.forward called:")
    print(f"    pre_computed_kv is None: {pre_computed_kv is None}")
    if pre_computed_kv is not None:
        k, v = pre_computed_kv
        print(f"    pre_computed_kv K shape: {k.shape}")
        print(f"    pre_computed_kv V shape: {v.shape}")
    print(f"    kv_storage_callback is None: {kv_storage_callback is None}")

    return original_forward(seqs, seqs_layout, keys, keys_layout, values, bias_cache, state_bag=state_bag, pre_computed_kv=pre_computed_kv, kv_storage_callback=kv_storage_callback)

layer20.self_attn.forward = traced_forward

input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
batch_layout = BatchLayout(input_ids.shape, [5], device=device)

kv_slots = {
    KVProjectionType.LOCAL: None,
    KVProjectionType.GLOBAL: None,
}

print("\nRunning forward through layers 0-20...")

with torch.no_grad():
    embeds, _ = model.decoder_frontend(input_ids, batch_layout, state_bag=None)
    hidden = model.decoder._stack_altup(embeds)

    attn_bias_cache = AttentionBiasCache()

    for idx in range(21):
        layer = model.decoder.layers[idx]

        if idx == 18:
            print(f"\nBefore layer 18 (LOCAL SOURCE):")
            print(f"  LOCAL slot: {kv_slots[KVProjectionType.LOCAL] is not None}")

        hidden = layer(
            hidden, batch_layout, attn_bias_cache,
            per_layer_input=None, state_bag=None,
            kv_projection_slots=kv_slots
        )

        if idx == 18:
            print(f"\nAfter layer 18:")
            if kv_slots[KVProjectionType.LOCAL] is not None:
                k, v = kv_slots[KVProjectionType.LOCAL]
                print(f"  LOCAL slot K shape: {k.shape}")
            else:
                print(f"  LOCAL slot is EMPTY!")

        if idx == 20:
            print(f"\nAfter layer 20 finished.")
            break

print("\nDone.")
