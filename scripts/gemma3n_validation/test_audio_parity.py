#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test audio tower parity between HuggingFace and fairseq2 implementations."""

import torch
from transformers import AutoModel

from fairseq2.models.gemma3n.audio_tower import Gemma3nAudioTower
from fairseq2.models.gemma3n.config import Gemma3nAudioConfig, get_gemma3n_e2b_config
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict


def load_hf_audio_tower():
    """Load HuggingFace Gemma3n model and extract audio tower."""
    print("Loading HuggingFace Gemma3n model...")
    hf_model = AutoModel.from_pretrained(
        "google/gemma-3n-E2B-it",
        trust_remote_code=True,
    )
    return hf_model.audio_tower, hf_model.embed_audio


def load_fs2_audio_tower(hf_state_dict):
    """Load fairseq2 audio tower with converted HF weights."""
    print("Converting checkpoint to fairseq2 format...")
    audio_config = Gemma3nAudioConfig()
    text_config = get_gemma3n_e2b_config()

    converted = convert_gemma3n_state_dict(hf_state_dict, text_config)

    # Extract audio tower weights
    audio_state = {
        k.replace("audio_tower.", ""): v
        for k, v in converted.items()
        if k.startswith("audio_tower.")
    }

    # Extract embedder weights
    embedder_state = {
        k.replace("embed_audio.", ""): v
        for k, v in converted.items()
        if k.startswith("embed_audio.")
    }

    print(f"  Audio tower parameters: {len(audio_state)}")
    print(f"  Embedder parameters: {len(embedder_state)}")

    # Create fairseq2 audio tower
    fs2_tower = Gemma3nAudioTower(audio_config, text_config)

    # Load weights
    print("Loading weights into fairseq2 audio tower...")

    # Load subsample weights
    subsample_state = {
        k.replace("subsample.", ""): v
        for k, v in audio_state.items()
        if k.startswith("subsample.")
    }
    missing, unexpected = fs2_tower.subsample.load_state_dict(subsample_state, strict=False)
    if missing:
        print(f"    ⚠ Missing subsample keys: {missing}")
    if unexpected:
        print(f"    ⚠ Unexpected subsample keys: {unexpected}")
    print(f"  ✓ Loaded subsample: {len(subsample_state)} parameters")

    # Load encoder weights
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in audio_state.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = fs2_tower.encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"    ⚠ Missing encoder keys: {missing}")
    if unexpected:
        print(f"    ⚠ Unexpected encoder keys: {unexpected}")
    print(f"  ✓ Loaded encoder: {len(encoder_state)} parameters")

    # Load embedder weights
    missing, unexpected = fs2_tower.embedder.load_state_dict(embedder_state, strict=False)
    if missing:
        print(f"    ⚠ Missing embedder keys: {missing}")
    if unexpected:
        print(f"    ⚠ Unexpected embedder keys: {unexpected}")
    print(f"  ✓ Loaded embedder: {len(embedder_state)} parameters")

    return fs2_tower


def test_audio_tower_parity():
    """Test that fairseq2 and HF audio towers produce identical outputs."""
    print("=" * 80)
    print("Audio Tower Parity Test")
    print("=" * 80)

    # Load models
    hf_audio_tower, hf_embedder = load_hf_audio_tower()
    hf_state = {**hf_audio_tower.state_dict(), **hf_embedder.state_dict()}

    # Prefix HF keys to match expected format for conversion
    prefixed_hf_state = {}
    for k, v in hf_audio_tower.state_dict().items():
        prefixed_hf_state[f"model.audio_tower.{k}"] = v
    for k, v in hf_embedder.state_dict().items():
        prefixed_hf_state[f"model.embed_audio.{k}"] = v

    fs2_tower = load_fs2_audio_tower(prefixed_hf_state)

    # Set both to eval mode
    hf_audio_tower.eval()
    hf_embedder.eval()
    fs2_tower.eval()

    print("\n" + "=" * 80)
    print("Running Parity Test")
    print("=" * 80)

    # Create synthetic mel-spectrogram input
    batch_size = 2
    time_steps = 100
    audio_config = Gemma3nAudioConfig()

    torch.manual_seed(42)
    mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)

    print(f"\nInput shape: {mel_features.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Time steps: {time_steps}")
    print(f"  Features: {audio_config.input_feat_size}")

    # Run HF forward pass
    print("\n--- HuggingFace Forward Pass ---")
    with torch.no_grad():
        # HF audio tower: subsample + conformer
        hf_conformer_out = hf_audio_tower(mel_features)
        print(f"After conformer: {hf_conformer_out.shape}")

        # HF embedder: project to text space
        hf_output = hf_embedder(hf_conformer_out)
        print(f"After embedder: {hf_output.shape}")

    # Run fairseq2 forward pass
    print("\n--- fairseq2 Forward Pass ---")
    with torch.no_grad():
        fs2_output, fs2_layout = fs2_tower(mel_features)
        print(f"Output: {fs2_output.shape}")
        print(f"Layout seq_lens: {fs2_layout.seq_lens}")

    # Compare outputs
    print("\n" + "=" * 80)
    print("Parity Results")
    print("=" * 80)

    # Check shapes match
    print(f"\nShape comparison:")
    print(f"  HF:  {hf_output.shape}")
    print(f"  FS2: {fs2_output.shape}")
    assert hf_output.shape == fs2_output.shape, "Output shapes don't match!"
    print("  ✓ Shapes match")

    # Compute differences
    abs_diff = (hf_output - fs2_output).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (hf_output.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\nNumerical differences:")
    print(f"  Max absolute diff: {max_abs_diff:.6e}")
    print(f"  Mean absolute diff: {mean_abs_diff:.6e}")
    print(f"  Max relative diff: {max_rel_diff:.6e}")
    print(f"  Mean relative diff: {mean_rel_diff:.6e}")

    # Check parity thresholds
    abs_threshold = 1e-4
    rel_threshold = 1e-3

    if max_abs_diff < abs_threshold and max_rel_diff < rel_threshold:
        print(f"\n✅ PARITY ACHIEVED!")
        print(f"   Max abs diff ({max_abs_diff:.6e}) < threshold ({abs_threshold})")
        print(f"   Max rel diff ({max_rel_diff:.6e}) < threshold ({rel_threshold})")
        return True
    else:
        print(f"\n❌ PARITY NOT ACHIEVED")
        if max_abs_diff >= abs_threshold:
            print(f"   Max abs diff ({max_abs_diff:.6e}) >= threshold ({abs_threshold})")
        if max_rel_diff >= rel_threshold:
            print(f"   Max rel diff ({max_rel_diff:.6e}) >= threshold ({rel_threshold})")

        # Show where biggest differences are
        max_idx = abs_diff.argmax()
        batch_idx = max_idx // (abs_diff.size(1) * abs_diff.size(2))
        time_idx = (max_idx // abs_diff.size(2)) % abs_diff.size(1)
        feat_idx = max_idx % abs_diff.size(2)

        print(f"\n   Largest diff at position [{batch_idx}, {time_idx}, {feat_idx}]:")
        print(f"     HF:  {hf_output[batch_idx, time_idx, feat_idx].item():.6f}")
        print(f"     FS2: {fs2_output[batch_idx, time_idx, feat_idx].item():.6f}")
        print(f"     Diff: {abs_diff[batch_idx, time_idx, feat_idx].item():.6e}")

        return False


if __name__ == "__main__":
    success = test_audio_tower_parity()
    exit(0 if success else 1)
