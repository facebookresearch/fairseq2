#!/usr/bin/env python3
"""Check if embeddings match after checkpoint loading."""

import torch
from transformers import AutoModelForCausalLM

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.factory import create_gemma3n_model
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict


def main() -> None:
    device = torch.device("cpu")

    # Load models
    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3n-E2B-it",
        torch_dtype=torch.float32,
        device_map=device,
        local_files_only=True,
    )

    config = get_gemma3n_e2b_config()
    fs2_model = create_gemma3n_model(config, device=device, dtype=torch.float32)

    # Load checkpoint
    hf_state_dict = hf_model.state_dict()
    fs2_state_dict = convert_gemma3n_state_dict(hf_state_dict, config)
    fs2_model.load_state_dict(fs2_state_dict, strict=False)

    # Compare embedding weights
    hf_embed = hf_model.model.language_model.embed_tokens.weight
    fs2_embed = fs2_model.decoder_frontend.embed.weight

    print("Embedding weights:")
    print(f"  HF shape:  {hf_embed.shape}")
    print(f"  FS2 shape: {fs2_embed.shape}")
    print(f"  Shapes match: {hf_embed.shape == fs2_embed.shape}")
    print(f"  Weights match: {torch.allclose(hf_embed, fs2_embed, atol=1e-6)}")
    print(f"  Max diff: {(hf_embed - fs2_embed).abs().max():.6e}")

    # Test embedding lookup
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    hf_embedded = hf_model.model.language_model.embed_tokens(token_ids)
    fs2_embedded = fs2_model.decoder_frontend.embed(token_ids)

    print("\nEmbedding lookup:")
    print(f"  HF result shape:  {hf_embedded.shape}")
    print(f"  FS2 result shape: {fs2_embedded.shape}")
    print(f"  Results match: {torch.allclose(hf_embedded, fs2_embedded, atol=1e-6)}")
    print(f"  Max diff: {(hf_embedded - fs2_embedded).abs().max():.6e}")

    # Check PLE embeddings
    hf_ple = hf_model.model.language_model.embed_tokens_per_layer.weight
    fs2_ple = fs2_model.decoder_frontend.embed_tokens_per_layer.weight

    print("\nPLE embedding weights:")
    print(f"  HF shape:  {hf_ple.shape}")
    print(f"  FS2 shape: {fs2_ple.shape}")
    print(f"  Shapes match: {hf_ple.shape == fs2_ple.shape}")
    print(f"  Weights match: {torch.allclose(hf_ple, fs2_ple, atol=1e-6)}")
    print(f"  Max diff: {(hf_ple - fs2_ple).abs().max():.6e}")


if __name__ == "__main__":
    main()
