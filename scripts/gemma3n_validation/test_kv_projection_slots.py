#!/usr/bin/env python3
"""Test KV projection slot mechanism with detailed tracing."""

from __future__ import annotations

import torch

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.nn import BatchLayout


def main() -> None:
    print("="*80)
    print("KV PROJECTION SLOT MECHANISM TEST")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create fairseq2 model
    print("\n[1/3] Creating fairseq2 model...")
    config = get_gemma3n_e2b_config()
    model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    model.eval()

    # Print KV projection configuration
    print("\n[2/3] KV Projection Configuration:")
    print("  Layer | Type   | Role     | Details")
    print("  " + "-"*60)

    for idx, layer in enumerate(model.decoder.layers):
        layer_type = "GLOBAL" if layer.is_global else "LOCAL"
        role = layer.kv_projection_role.value

        details = ""
        if role == "source":
            details = f"stores to {layer_type} slot"
        elif role == "consumer":
            details = f"reads from {layer_type} slot"

        print(f"  {idx:5d} | {layer_type:6s} | {role:8s} | {details}")

    # Prepare test input
    print("\n[3/3] Running forward pass with slot tracing...")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    seq_lens = [input_ids.shape[1]]
    batch_layout = BatchLayout(input_ids.shape, seq_lens, device=device)

    # Patch decoder to trace slot access
    original_forward = model.decoder.forward

    def traced_forward(seqs, seqs_layout, *, state_bag=None):
        print("\n  Decoder creating KV projection slots...")

        # Call original but trace slot updates
        from fairseq2.models.gemma3n.kv_projection import KVProjectionType

        kv_projection_slots = None
        if model.decoder._has_kv_projection_sharing:
            kv_projection_slots = {
                KVProjectionType.LOCAL: None,
                KVProjectionType.GLOBAL: None,
            }
            print(f"    Created slots: LOCAL=None, GLOBAL=None")

        # Stack to 4D
        hidden_states = model.decoder._stack_altup(seqs)

        # Get PLE embeddings
        per_layer_inputs = getattr(state_bag, 'per_layer_inputs', None)

        # Process through layers with tracing
        from fairseq2.models.transformer import AttentionBiasCache
        attn_bias_cache = AttentionBiasCache()

        for layer_idx, layer in enumerate(model.decoder.layers):
            if per_layer_inputs is not None:
                layer_ple = per_layer_inputs[:, :, layer_idx, :]
            else:
                layer_ple = None

            # Trace slot state before layer
            if kv_projection_slots is not None:
                from fairseq2.models.gemma3n.kv_projection import KVProjectionRole, KVProjectionType

                local_status = "populated" if kv_projection_slots[KVProjectionType.LOCAL] is not None else "empty"
                global_status = "populated" if kv_projection_slots[KVProjectionType.GLOBAL] is not None else "empty"

                role = layer.kv_projection_role
                layer_type = "GLOBAL" if layer.is_global else "LOCAL"

                if role == KVProjectionRole.SOURCE:
                    print(f"\n  Layer {layer_idx} ({layer_type}, SOURCE):")
                    print(f"    Before: LOCAL={local_status}, GLOBAL={global_status}")
                    print(f"    Will STORE to {layer_type} slot")
                elif role == KVProjectionRole.CONSUMER:
                    print(f"\n  Layer {layer_idx} ({layer_type}, CONSUMER):")
                    print(f"    Before: LOCAL={local_status}, GLOBAL={global_status}")
                    print(f"    Will READ from {layer_type} slot")

            # Process layer
            hidden_states = layer(
                hidden_states, seqs_layout, attn_bias_cache,
                per_layer_input=layer_ple,
                state_bag=state_bag,
                kv_projection_slots=kv_projection_slots,
            )

            # Trace slot state after layer
            if kv_projection_slots is not None:
                local_status = "populated" if kv_projection_slots[KVProjectionType.LOCAL] is not None else "empty"
                global_status = "populated" if kv_projection_slots[KVProjectionType.GLOBAL] is not None else "empty"

                if role in [KVProjectionRole.SOURCE, KVProjectionRole.CONSUMER]:
                    print(f"    After:  LOCAL={local_status}, GLOBAL={global_status}")

        # Unstack and normalize
        seqs = model.decoder._unstack_altup(hidden_states)
        seqs = model.decoder.layer_norm(seqs)
        return seqs

    model.decoder.forward = traced_forward

    with torch.no_grad():
        logits = model(input_ids, batch_layout)

    print("\n" + "="*80)
    print("TEST COMPLETED")
    print(f"Output shape: {logits.shape}")
    print("="*80)


if __name__ == "__main__":
    main()
