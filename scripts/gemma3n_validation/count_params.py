#!/usr/bin/env python3
"""Count parameters by component to debug size mismatch."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model


def count_params_by_prefix(model, prefixes: list[str]) -> dict[str, int]:
    """Count parameters grouped by prefix."""
    counts = {prefix: 0 for prefix in prefixes}
    counts["other"] = 0

    for name, param in model.named_parameters():
        matched = False
        for prefix in prefixes:
            if name.startswith(prefix):
                counts[prefix] += param.numel()
                matched = True
                break
        if not matched:
            counts["other"] += param.numel()

    return counts


def main() -> None:
    print("="*80)
    print("PARAMETER COUNT COMPARISON")
    print("="*80)

    device = torch.device("cpu")

    # Load HF model
    print("\n[1/2] Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E2B-it",
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    hf_total = sum(p.numel() for p in hf_model.parameters())
    print(f"✓ Total HF parameters: {hf_total:,}")

    # Count HF params by component
    hf_prefixes = [
        "model.language_model.embed_tokens",
        "model.language_model.embed_tokens_per_layer",
        "model.language_model.per_layer_model_projection",
        "model.language_model.altup_projections",
        "model.language_model.altup_unembed_projections",
        "model.language_model.layers",
        "model.language_model.final_norm",
        "model.lm_head",
    ]
    hf_counts = count_params_by_prefix(hf_model, hf_prefixes)

    print("\nHuggingFace parameter breakdown:")
    for prefix, count in sorted(hf_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / hf_total * 100
            print(f"  {prefix:50s}: {count:12,} ({pct:5.2f}%)")

    # Create FS2 model
    print("\n[2/2] Creating fairseq2 model...")
    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_total = sum(p.numel() for p in fs2_model.parameters())
    print(f"✓ Total FS2 parameters: {fs2_total:,}")

    # Count FS2 params by component
    fs2_prefixes = [
        "decoder_frontend.embed",
        "decoder_frontend.embed_tokens_per_layer",
        "decoder_frontend.per_layer_model_projection",
        "decoder.altup_projections",
        "decoder.altup_unembed_projections",
        "decoder.layers",
        "decoder.layer_norm",
        "final_proj",
    ]
    fs2_counts = count_params_by_prefix(fs2_model, fs2_prefixes)

    print("\nfairseq2 parameter breakdown:")
    for prefix, count in sorted(fs2_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / fs2_total * 100
            print(f"  {prefix:50s}: {count:12,} ({pct:5.2f}%)")

    # Compare
    print("\n" + "="*80)
    print(f"Difference: {hf_total - fs2_total:,} parameters")
    print(f"  HF:  {hf_total:,}")
    print(f"  FS2: {fs2_total:,}")
    print("="*80)


if __name__ == "__main__":
    main()
