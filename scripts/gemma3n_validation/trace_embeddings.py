#!/usr/bin/env python3
"""Trace embedding forward pass in detail."""

import torch
from transformers import AutoModelForCausalLM

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict
from fairseq2.nn import BatchLayout, IncrementalStateBag


def main() -> None:
    device = torch.device("cpu")

    # Load models
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E2B-it",
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )
    hf_model.eval()

    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)
    fs2_model.eval()

    # Load checkpoint
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    fs2_model.load_state_dict(fs2_state_dict, strict=False)

    # Test input
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])

    print("="*80)
    print("HuggingFace Embedding Forward")
    print("="*80)

    # HF embedding lookup (raw)
    hf_raw_embeds = hf_model.model.language_model.embed_tokens(token_ids)
    print(f"\nRaw embed_tokens output:")
    print(f"  shape: {hf_raw_embeds.shape}")
    print(f"  mean: {hf_raw_embeds.mean():.6f}")
    print(f"  std: {hf_raw_embeds.std():.6f}")
    print(f"  first token first 5 dims: {hf_raw_embeds[0, 0, :5].tolist()}")

    print("\n" + "="*80)
    print("fairseq2 Embedding Forward")
    print("="*80)

    # FS2 raw embedding lookup
    fs2_raw_embeds = fs2_model.decoder_frontend.embed(token_ids)
    print(f"\nRaw embed output:")
    print(f"  shape: {fs2_raw_embeds.shape}")
    print(f"  mean: {fs2_raw_embeds.mean():.6f}")
    print(f"  std: {fs2_raw_embeds.std():.6f}")
    print(f"  first token first 5 dims: {fs2_raw_embeds[0, 0, :5].tolist()}")

    # Check scale attribute
    print(f"\nFrontend scale: {fs2_model.decoder_frontend.scale}")

    # Full frontend forward
    batch_layout = BatchLayout(token_ids.shape, [token_ids.shape[1]], device=device)
    state_bag = IncrementalStateBag(max_num_steps=token_ids.size(1))

    fs2_frontend_out, _ = fs2_model.decoder_frontend(token_ids, batch_layout, state_bag=state_bag)
    print(f"\nFull frontend output:")
    print(f"  shape: {fs2_frontend_out.shape}")
    print(f"  mean: {fs2_frontend_out.mean():.6f}")
    print(f"  std: {fs2_frontend_out.std():.6f}")
    print(f"  first token first 5 dims: {fs2_frontend_out[0, 0, :5].tolist()}")

    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    # Raw embeddings diff
    raw_diff = (hf_raw_embeds - fs2_raw_embeds).abs()
    print(f"\nRaw embeddings diff:")
    print(f"  max: {raw_diff.max():.6e}")
    print(f"  mean: {raw_diff.mean():.6e}")
    print(f"  match: {torch.allclose(hf_raw_embeds, fs2_raw_embeds, atol=1e-6)}")

    # Frontend output diff
    frontend_diff = (hf_raw_embeds - fs2_frontend_out).abs()
    print(f"\nHF raw embeds vs FS2 frontend output diff:")
    print(f"  max: {frontend_diff.max():.6e}")
    print(f"  mean: {frontend_diff.mean():.6e}")


if __name__ == "__main__":
    main()
