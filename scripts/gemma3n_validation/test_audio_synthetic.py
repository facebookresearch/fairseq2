#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test audio tower architecture with synthetic weights (no HF download required)."""

import torch

from fairseq2.models.gemma3n.audio_tower import Gemma3nAudioTower
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, get_gemma3n_e2b_config


def test_audio_tower_synthetic():
    """Test audio tower with synthetic weights to verify architecture."""
    print("=" * 80)
    print("Audio Tower Synthetic Test (No Network Required)")
    print("=" * 80)

    audio_config = Gemma3nAudioConfig()
    text_config = get_gemma3n_e2b_config()

    # Create audio tower
    print("\nCreating fairseq2 audio tower...")
    tower = Gemma3nAudioTower(audio_config, text_config)
    tower.eval()

    total_params = sum(p.numel() for p in tower.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create synthetic input
    batch_size = 2
    time_steps = 100

    print(f"\nInput configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time steps: {time_steps}")
    print(f"  Features: {audio_config.input_feat_size}")

    torch.manual_seed(42)
    mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

    # Run forward pass
    print("\n--- Forward Pass ---")
    with torch.no_grad():
        output, layout = tower(mel_features)

    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {time_steps//4}, {text_config.model_dim})")
    print(f"  Layout seq_lens: {layout.seq_lens}")

    # Verify output shape
    expected_time = time_steps // 4
    assert output.shape == (batch_size, expected_time, text_config.model_dim)
    print("  ✓ Output shape correct")

    # Verify layout
    assert len(layout.seq_lens) == batch_size
    assert all(l == expected_time for l in layout.seq_lens)
    print("  ✓ Layout correct")

    # Verify output is not NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print("  ✓ Output is finite")

    # Test different input sizes
    print("\n--- Testing Variable Input Sizes ---")
    for test_time_steps in [40, 80, 120, 160]:
        test_input = torch.randn(1, test_time_steps, audio_config.input_feat_size)
        with torch.no_grad():
            test_output, test_layout = tower(test_input)
        expected = test_time_steps // 4
        assert test_output.size(1) == expected
        print(f"  ✓ Input {test_time_steps} → Output {expected}")

    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    tower.train()
    test_input = torch.randn(2, 100, audio_config.input_feat_size, requires_grad=True)
    test_output, _ = tower(test_input)
    loss = test_output.sum()
    loss.backward()

    # Check gradients exist
    has_grad = test_input.grad is not None
    print(f"  Input gradient: {'✓ Present' if has_grad else '✗ Missing'}")

    param_with_grad = sum(1 for p in tower.parameters() if p.grad is not None)
    total_param_count = sum(1 for _ in tower.parameters())
    print(f"  Parameters with gradients: {param_with_grad}/{total_param_count}")

    if param_with_grad == total_param_count:
        print("  ✓ All parameters have gradients")
    else:
        print(f"  ⚠ Warning: {total_param_count - param_with_grad} parameters missing gradients")

    # Test individual components
    print("\n--- Component Verification ---")

    # Subsample
    subsample_out = tower.subsample(mel_features)
    print(f"  Subsample output: {subsample_out.shape}")
    assert subsample_out.shape == (batch_size, time_steps//4, audio_config.hidden_size)
    print("    ✓ Shape correct")

    # Encoder
    from fairseq2.nn import BatchLayout as BL
    encoder_layout = BL((batch_size, subsample_out.size(1)), seq_lens=[subsample_out.size(1)]*batch_size)
    encoder_out = tower.encoder(subsample_out, encoder_layout)
    print(f"  Encoder output: {encoder_out.shape}")
    assert encoder_out.shape == subsample_out.shape
    print("    ✓ Shape correct")

    # Embedder
    embedder_out = tower.embedder(encoder_out, is_soft=True)
    print(f"  Embedder output: {embedder_out.shape}")
    assert embedder_out.shape == (batch_size, time_steps//4, text_config.model_dim)
    print("    ✓ Shape correct")

    print("\n" + "=" * 80)
    print("✅ All Synthetic Tests Passed!")
    print("=" * 80)
    print("\nArchitecture verification complete.")
    print("Ready for parity testing with real HuggingFace weights.")


if __name__ == "__main__":
    test_audio_tower_synthetic()
