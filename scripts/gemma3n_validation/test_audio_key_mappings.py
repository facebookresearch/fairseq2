#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test audio tower checkpoint key mappings."""

import torch

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict


def test_audio_key_mappings():
    """Test that audio tower HF keys are correctly mapped to fairseq2 keys."""
    print("Testing audio tower key mappings...\n")

    # Create mock HF state dict with audio tower weights
    hf_state_dict = {}

    # Subsample projection
    subsample_keys = {
        "model.audio_tower.subsample_conv_projection.conv_0.conv.weight": (128, 1, 3, 3),
        "model.audio_tower.subsample_conv_projection.conv_0.norm.weight": (128,),
        "model.audio_tower.subsample_conv_projection.conv_1.conv.weight": (32, 128, 3, 3),
        "model.audio_tower.subsample_conv_projection.conv_1.norm.weight": (32,),
        "model.audio_tower.subsample_conv_projection.input_proj_linear.weight": (1536, 1024),
    }

    # Conformer layer 0 - attention
    conformer_attn_keys = {
        "model.audio_tower.conformer.0.attention.pre_attn_norm.weight": (1536,),
        "model.audio_tower.conformer.0.attention.attn.q_proj.weight": (1536, 1536),
        "model.audio_tower.conformer.0.attention.attn.k_proj.weight": (1536, 1536),
        "model.audio_tower.conformer.0.attention.attn.v_proj.weight": (1536, 1536),
        "model.audio_tower.conformer.0.attention.attn.per_dim_scale": (192,),
        "model.audio_tower.conformer.0.attention.attn.relative_position_embedding.pos_proj.weight": (1536, 1536),
        "model.audio_tower.conformer.0.attention.post.weight": (1536, 1536),
    }

    # Conformer layer 0 - FFN start
    conformer_ffn1_keys = {
        "model.audio_tower.conformer.0.ffw_layer_start.pre_layer_norm.weight": (1536,),
        "model.audio_tower.conformer.0.ffw_layer_start.ffw_layer_1.weight": (6144, 1536),
        "model.audio_tower.conformer.0.ffw_layer_start.ffw_layer_2.weight": (1536, 6144),
    }

    # Conformer layer 0 - Conv
    conformer_conv_keys = {
        "model.audio_tower.conformer.0.lconv1d.pre_layer_norm.weight": (1536,),
        "model.audio_tower.conformer.0.lconv1d.linear_start.weight": (3072, 1536),
        "model.audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight": (1536, 1, 5),
        "model.audio_tower.conformer.0.lconv1d.conv_norm.weight": (1536,),
        "model.audio_tower.conformer.0.lconv1d.linear_end.weight": (1536, 1536),
    }

    # Conformer layer 0 - FFN end
    conformer_ffn2_keys = {
        "model.audio_tower.conformer.0.ffw_layer_end.pre_layer_norm.weight": (1536,),
        "model.audio_tower.conformer.0.ffw_layer_end.ffw_layer_1.weight": (6144, 1536),
        "model.audio_tower.conformer.0.ffw_layer_end.ffw_layer_2.weight": (1536, 6144),
    }

    # Conformer layer 0 - Final norm
    conformer_norm_keys = {
        "model.audio_tower.conformer.0.norm.weight": (1536,),
    }

    # Multimodal embedder
    embedder_keys = {
        "model.embed_audio.embedding.weight": (128, 1536),
        "model.embed_audio.hard_embedding_norm.weight": (1536,),
        "model.embed_audio.soft_embedding_norm.weight": (1536,),
        "model.embed_audio.embedding_projection.weight": (2048, 1536),
        "model.embed_audio.embedding_post_projection_norm.weight": (2048,),
    }

    # Add minimal text model keys
    text_keys = {
        "model.language_model.embed_tokens.weight": (262400, 2048),
        "lm_head.weight": (262400, 2048),
    }

    # Combine all keys
    all_keys = {
        **subsample_keys,
        **conformer_attn_keys,
        **conformer_ffn1_keys,
        **conformer_conv_keys,
        **conformer_ffn2_keys,
        **conformer_norm_keys,
        **embedder_keys,
        **text_keys,
    }

    # Create tensors
    for key, shape in all_keys.items():
        hf_state_dict[key] = torch.randn(shape)

    print(f"Created mock HF state dict with {len(hf_state_dict)} tensors")

    # Convert
    config = get_gemma3n_e2b_config()
    converted = convert_gemma3n_state_dict(hf_state_dict, config)

    print(f"Converted to fairseq2 format: {len(converted)} tensors\n")

    # Test subsample mappings
    print("✓ Testing subsample projection mappings...")
    assert "audio_tower.subsample.conv_0.weight" in converted
    assert "audio_tower.subsample.norm_0.weight" in converted
    assert "audio_tower.subsample.proj.weight" in converted
    assert converted["audio_tower.subsample.conv_0.weight"].shape == torch.Size([128, 1, 3, 3])
    print("  All subsample keys mapped correctly")

    # Test conformer attention mappings
    print("\n✓ Testing conformer attention mappings...")
    assert "audio_tower.encoder.layers.0.self_attn_layer_norm.weight" in converted
    assert "audio_tower.encoder.layers.0.self_attn.q_proj.weight" in converted
    assert "audio_tower.encoder.layers.0.self_attn.sdpa.per_dim_scale" in converted
    assert "audio_tower.encoder.layers.0.self_attn.sdpa.rel_k_embed.weight" in converted
    assert "audio_tower.encoder.layers.0.self_attn.output_proj.weight" in converted
    assert converted["audio_tower.encoder.layers.0.self_attn.sdpa.per_dim_scale"].shape == torch.Size([192])
    print("  All attention keys mapped correctly")

    # Test FFN mappings
    print("\n✓ Testing FFN mappings...")
    assert "audio_tower.encoder.layers.0.ffn1_layer_norm.weight" in converted
    assert "audio_tower.encoder.layers.0.ffn1.inner_proj.weight" in converted
    assert "audio_tower.encoder.layers.0.ffn2.output_proj.weight" in converted
    print("  All FFN keys mapped correctly")

    # Test conv mappings
    print("\n✓ Testing convolution mappings...")
    assert "audio_tower.encoder.layers.0.conv_layer_norm.weight" in converted
    assert "audio_tower.encoder.layers.0.conv.pointwise_conv1.weight" in converted
    assert "audio_tower.encoder.layers.0.conv.depthwise_conv.weight" in converted
    assert "audio_tower.encoder.layers.0.conv.layer_norm.weight" in converted
    assert "audio_tower.encoder.layers.0.conv.pointwise_conv2.weight" in converted
    print("  All conv keys mapped correctly")

    # Test layer norm mapping
    print("\n✓ Testing final norm mapping...")
    assert "audio_tower.encoder.layers.0.layer_norm.weight" in converted
    print("  Final norm key mapped correctly")

    # Test embedder mappings
    print("\n✓ Testing multimodal embedder mappings...")
    assert "embed_audio.embedding.weight" in converted
    assert "embed_audio.hard_embedding_norm.weight" in converted
    assert "embed_audio.soft_embedding_norm.weight" in converted
    assert "embed_audio.embedding_projection.weight" in converted
    assert "embed_audio.embedding_post_projection_norm.weight" in converted
    assert converted["embed_audio.embedding.weight"].shape == torch.Size([128, 1536])
    assert converted["embed_audio.embedding_projection.weight"].shape == torch.Size([2048, 1536])
    print("  All embedder keys mapped correctly")

    # Test text model mappings
    print("\n✓ Testing text model mappings...")
    assert "decoder_frontend.embed.weight" in converted
    assert "final_proj.weight" in converted
    print("  Text model keys mapped correctly")

    # Count audio tower parameters
    audio_keys = [k for k in converted.keys() if "audio_tower" in k or "embed_audio" in k]
    print(f"\n✅ All audio tower key mappings validated!")
    print(f"   Audio tower parameters: {len(audio_keys)}")
    print(f"   Text model parameters: {len(converted) - len(audio_keys)}")


if __name__ == "__main__":
    test_audio_key_mappings()
