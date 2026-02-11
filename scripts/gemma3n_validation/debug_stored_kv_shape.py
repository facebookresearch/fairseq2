#!/usr/bin/env python3
"""Debug: Print shapes of stored K/V in layer 18."""

from __future__ import annotations

import torch
from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.kv_projection import KVProjectionType
from fairseq2.models.transformer import AttentionBiasCache
from fairseq2.nn import BatchLayout

device = torch.device("cpu")
dtype = torch.float32

config = get_gemma3n_e2b_config()
model = create_gemma3n_model(config, device=device, dtype=dtype).eval()

input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
batch_layout = BatchLayout(input_ids.shape, [5], device=device)

# Create slots
kv_slots = {
    KVProjectionType.LOCAL: None,
    KVProjectionType.GLOBAL: None,
}

print("Running forward through layer 18 (LOCAL SOURCE)...")

with torch.no_grad():
    # Get embeddings
    embeds, _ = model.decoder_frontend(input_ids, batch_layout, state_bag=None)
    hidden = model.decoder._stack_altup(embeds)

    attn_bias_cache = AttentionBiasCache()

    # Run through layers 0-18
    for idx in range(19):
        layer = model.decoder.layers[idx]
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
                print(f"  LOCAL slot V shape: {v.shape}")
                print(f"  K dtype: {k.dtype}")
                print(f"  K device: {k.device}")
                print(f"  K min/max: {k.min().item():.6f} / {k.max().item():.6f}")
            else:
                print(f"  LOCAL slot is EMPTY!")

    print(f"\nExpected shape: (batch=1, seq_len=5, num_kv_heads=2, head_dim=256)")
    print(f"Note: Gemma3n has num_key_value_heads=2 (GQA)")
