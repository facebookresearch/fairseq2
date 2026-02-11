#!/usr/bin/env python3
"""Debug checkpoint loading by inspecting actual weight values."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict


def main() -> None:
    print("="*80)
    print("CHECKPOINT LOADING DEBUG")
    print("="*80)

    device = torch.device("cpu")

    # Load HF model
    print("\n[1/3] Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E2B-it",
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    hf_model.eval()

    # Create FS2 model
    print("\n[2/3] Creating fairseq2 model...")
    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()

    # Sample some weights BEFORE loading
    print("\n[3/3] Checking weight loading...")

    # Get a sample HF weight
    hf_embed_weight = hf_model.model.language_model.embed_tokens.weight
    hf_layer0_q = hf_model.model.language_model.layers[0].self_attn.q_proj.weight

    print(f"\nHF weights:")
    print(f"  embed_tokens.weight: shape={hf_embed_weight.shape}, mean={hf_embed_weight.mean():.6f}, std={hf_embed_weight.std():.6f}")
    print(f"  layer[0].q_proj.weight: shape={hf_layer0_q.shape}, mean={hf_layer0_q.mean():.6f}, std={hf_layer0_q.std():.6f}")

    # FS2 BEFORE loading
    fs2_embed_weight_before = fs2_model.decoder_frontend.embed.weight
    fs2_layer0_q_before = fs2_model.decoder.layers[0].self_attn.q_proj.weight

    print(f"\nFS2 weights BEFORE loading:")
    print(f"  decoder_frontend.embed.weight: shape={fs2_embed_weight_before.shape}, mean={fs2_embed_weight_before.mean():.6f}, std={fs2_embed_weight_before.std():.6f}")
    print(f"  decoder.layers[0].self_attn.q_proj.weight: shape={fs2_layer0_q_before.shape}, mean={fs2_layer0_q_before.mean():.6f}, std={fs2_layer0_q_before.std():.6f}")

    # Convert and load
    print("\n[Converting checkpoint...]")
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)

    # Check if keys exist in converted dict
    print(f"\nConverted state dict contains:")
    print(f"  'decoder_frontend.embed.weight': {'decoder_frontend.embed.weight' in fs2_state_dict}")
    print(f"  'decoder.layers.0.self_attn.q_proj.weight': {'decoder.layers.0.self_attn.q_proj.weight' in fs2_state_dict}")

    if 'decoder_frontend.embed.weight' in fs2_state_dict:
        converted_embed = fs2_state_dict['decoder_frontend.embed.weight']
        print(f"\nConverted embed stats: mean={converted_embed.mean():.6f}, std={converted_embed.std():.6f}")
        print(f"  Matches HF embed: {torch.allclose(converted_embed, hf_embed_weight, atol=1e-5)}")

    if 'decoder.layers.0.self_attn.q_proj.weight' in fs2_state_dict:
        converted_q = fs2_state_dict['decoder.layers.0.self_attn.q_proj.weight']
        print(f"\nConverted layer0.q_proj stats: mean={converted_q.mean():.6f}, std={converted_q.std():.6f}")
        print(f"  Matches HF layer0.q_proj: {torch.allclose(converted_q, hf_layer0_q, atol=1e-5)}")

    print("\n[Loading state dict...]")
    missing, unexpected = fs2_model.load_state_dict(fs2_state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # FS2 AFTER loading
    fs2_embed_weight_after = fs2_model.decoder_frontend.embed.weight
    fs2_layer0_q_after = fs2_model.decoder.layers[0].self_attn.q_proj.weight

    print(f"\nFS2 weights AFTER loading:")
    print(f"  decoder_frontend.embed.weight: shape={fs2_embed_weight_after.shape}, mean={fs2_embed_weight_after.mean():.6f}, std={fs2_embed_weight_after.std():.6f}")
    print(f"  decoder.layers[0].self_attn.q_proj.weight: shape={fs2_layer0_q_after.shape}, mean={fs2_layer0_q_after.mean():.6f}, std={fs2_layer0_q_after.std():.6f}")

    # Check if weights actually changed
    print(f"\nWeights changed after loading:")
    print(f"  embed.weight: {not torch.equal(fs2_embed_weight_before, fs2_embed_weight_after)}")
    print(f"  layer[0].q_proj.weight: {not torch.equal(fs2_layer0_q_before, fs2_layer0_q_after)}")

    # Check if FS2 matches HF after loading
    print(f"\nFS2 matches HF after loading:")
    print(f"  embed.weight: {torch.allclose(fs2_embed_weight_after, hf_embed_weight, atol=1e-5)}")
    print(f"  layer[0].q_proj.weight: {torch.allclose(fs2_layer0_q_after, hf_layer0_q, atol=1e-5)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
