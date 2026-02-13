#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test audio tower checkpoint conversion from HuggingFace to fairseq2."""

from transformers import AutoModel

from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, get_gemma3n_e2b_config
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict


def test_audio_checkpoint_conversion():
    """Load HF checkpoint and verify audio tower weights are converted correctly."""
    print("Loading HuggingFace Gemma3n model...")
    hf_model = AutoModel.from_pretrained(
        "google/gemma-3n-E2B-it",
        trust_remote_code=True,
    )

    print("Converting state dict...")
    config = get_gemma3n_e2b_config()
    audio_config = Gemma3nAudioConfig()

    converted = convert_gemma3n_state_dict(hf_model.state_dict(), config)

    # Check subsample projection weights
    subsample_keys = [k for k in converted.keys() if k.startswith("audio_tower.subsample.")]
    print(f"\n✓ Subsample projection: {len(subsample_keys)} parameters")
    expected_subsample = {
        "audio_tower.subsample.conv_0.weight",
        "audio_tower.subsample.conv_0.bias",
        "audio_tower.subsample.norm_0.weight",
        "audio_tower.subsample.norm_0.bias",
        "audio_tower.subsample.conv_1.weight",
        "audio_tower.subsample.conv_1.bias",
        "audio_tower.subsample.norm_1.weight",
        "audio_tower.subsample.norm_1.bias",
        "audio_tower.subsample.proj.weight",
        "audio_tower.subsample.proj.bias",
    }
    for key in expected_subsample:
        assert key in converted, f"Missing subsample key: {key}"
    print(f"  All {len(expected_subsample)} expected keys present")

    # Check conformer encoder weights
    encoder_keys = [k for k in converted.keys() if k.startswith("audio_tower.encoder.")]
    print(f"\n✓ Conformer encoder: {len(encoder_keys)} parameters")

    # Verify all 12 layers present
    for layer_idx in range(audio_config.conf_num_hidden_layers):
        layer_keys = [k for k in encoder_keys if k.startswith(f"audio_tower.encoder.layers.{layer_idx}.")]
        assert len(layer_keys) > 0, f"Missing layer {layer_idx}"
    print(f"  All {audio_config.conf_num_hidden_layers} conformer layers present")

    # Check key components in layer 0
    layer0_keys = [k for k in encoder_keys if k.startswith("audio_tower.encoder.layers.0.")]
    expected_components = ["self_attn", "ffn1", "ffn2", "conv", "layer_norm"]
    for component in expected_components:
        component_keys = [k for k in layer0_keys if component in k]
        assert len(component_keys) > 0, f"Missing component: {component}"
    print(f"  All key components present in layer 0")

    # Check per-dim scale and relative position embeddings
    assert "audio_tower.encoder.layers.0.self_attn.sdpa.per_dim_scale" in converted
    assert "audio_tower.encoder.layers.0.self_attn.sdpa.rel_k_embed.weight" in converted
    print(f"  ✓ Shaw relative position embeddings present")
    print(f"  ✓ Per-dimension scaling present")

    # Check multimodal embedder weights
    embedder_keys = [k for k in converted.keys() if k.startswith("embed_audio.")]
    print(f"\n✓ Multimodal embedder: {len(embedder_keys)} parameters")
    expected_embedder = {
        "embed_audio.embedding.weight",
        "embed_audio.hard_embedding_norm.weight",
        "embed_audio.hard_embedding_norm.bias",
        "embed_audio.soft_embedding_norm.weight",
        "embed_audio.soft_embedding_norm.bias",
        "embed_audio.embedding_projection.weight",
        "embed_audio.embedding_post_projection_norm.weight",
    }
    for key in expected_embedder:
        assert key in converted, f"Missing embedder key: {key}"
    print(f"  All {len(expected_embedder)} expected keys present")

    # Check shapes
    print("\n✓ Checking key tensor shapes...")
    import torch
    assert converted["audio_tower.subsample.conv_0.weight"].shape == torch.Size([128, 1, 3, 3])
    assert converted["audio_tower.subsample.proj.weight"].shape == torch.Size([1536, 1024])
    assert converted["audio_tower.encoder.layers.0.self_attn.sdpa.per_dim_scale"].shape == torch.Size([192])
    assert converted["embed_audio.embedding.weight"].shape == torch.Size([128, 1536])
    assert converted["embed_audio.embedding_projection.weight"].shape == torch.Size([2048, 1536])
    print("  All shapes correct")

    # Verify vision tower is filtered out
    vision_keys = [k for k in converted.keys() if "vision" in k.lower()]
    assert len(vision_keys) == 0, "Vision tower should be filtered out"
    print("\n✓ Vision tower correctly filtered out")

    total_audio_keys = len(subsample_keys) + len(encoder_keys) + len(embedder_keys)
    print(f"\n✅ Audio tower checkpoint conversion successful!")
    print(f"   Total audio parameters: {total_audio_keys}")


if __name__ == "__main__":
    test_audio_checkpoint_conversion()
