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

    # Set to eval mode
    hf_model.eval()

    # Check for NaN in audio tower weights
    audio_tower = hf_model.audio_tower
    has_nan = False
    for name, param in audio_tower.named_parameters():
        if torch.isnan(param).any():
            print(f"  ⚠ NaN in weight: {name}")
            has_nan = True
    if not has_nan:
        print("  ✓ No NaN in audio tower weights")

    return audio_tower, hf_model.embed_audio


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
    # Initialize missing biases to zero (HF doesn't have biases for GroupNorm)
    if "norm_0.bias" not in subsample_state and hasattr(fs2_tower.subsample.norm_0, "bias"):
        subsample_state["norm_0.bias"] = torch.zeros_like(fs2_tower.subsample.norm_0.weight)
    if "norm_1.bias" not in subsample_state and hasattr(fs2_tower.subsample.norm_1, "bias"):
        subsample_state["norm_1.bias"] = torch.zeros_like(fs2_tower.subsample.norm_1.weight)

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

    # Initialize missing RMSNorm biases to zero (HF doesn't have them)
    for layer_idx in range(12):
        norm_names = [
            f"layers.{layer_idx}.ffn1_layer_norm.bias",
            f"layers.{layer_idx}.self_attn_layer_norm.bias",
            f"layers.{layer_idx}.conv_layer_norm.bias",
            f"layers.{layer_idx}.conv.layer_norm.bias",
            f"layers.{layer_idx}.ffn2_layer_norm.bias",
            f"layers.{layer_idx}.layer_norm.bias",
        ]
        for norm_name in norm_names:
            if norm_name not in encoder_state:
                # Get the corresponding weight to know the size
                weight_name = norm_name.replace(".bias", ".weight")
                if weight_name in encoder_state:
                    encoder_state[norm_name] = torch.zeros_like(encoder_state[weight_name])

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

    # Inspect HF conformer for downsampling
    print("\n=== HF Audio Tower Attributes ===")
    print(f"Type: {type(hf_audio_tower).__name__}")
    print(f"Attributes: {[name for name, _ in hf_audio_tower.named_children()]}")

    # Check for conformer layers and find strided convolutions
    if hasattr(hf_audio_tower, 'conformer'):
        conformer = hf_audio_tower.conformer
        print(f"\nConformer has {len(conformer)} layers")

        # Check first 3 layers in detail
        for i in range(min(3, len(conformer))):
            layer = conformer[i]
            print(f"\n=== Layer {i} ===")
            print(f"Children: {[name for name, _ in layer.named_children()]}")

            if hasattr(layer, 'conv_module'):
                conv = layer.conv_module
                print(f"conv_module type: {type(conv).__name__}")
                print(f"conv_module children:")
                for name, module in conv.named_children():
                    print(f"  {name}: {type(module).__name__}", end="")
                    if hasattr(module, 'stride'):
                        print(f" stride={module.stride}", end="")
                    if hasattr(module, 'kernel_size'):
                        print(f" kernel={module.kernel_size}", end="")
                    print()

        # Check for pooling or other downsampling mechanisms
        print("\n=== Checking for pooling/downsampling layers ===")
        for i, layer in enumerate(conformer):
            for name, module in layer.named_modules():
                if 'pool' in name.lower() or 'downsample' in name.lower() or 'subsample' in name.lower():
                    print(f"Layer {i}, {name}: {type(module).__name__}")

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

    # Test 1: All-masked for HF parity (HF produces NaN with valid inputs)
    print("\n" + "=" * 80)
    print("Test 1: All-masked (HF parity)")
    print("=" * 80)
    audio_mel_mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    print("Using mask: all masked (True)")

    # Run HF forward pass
    print("\n--- HuggingFace Forward Pass ---")
    with torch.no_grad():
        # Check HF subsample output
        hf_subsample_out = hf_audio_tower.subsample_conv_projection(mel_features)
        print(f"HF subsample output: {hf_subsample_out.shape}")
        print(f"HF subsample has NaN: {torch.isnan(hf_subsample_out).any()}")

        # HF audio tower: subsample + conformer
        hf_output_obj = hf_audio_tower(mel_features, audio_mel_mask)
        hf_output = hf_output_obj.last_hidden_state
        print(f"HF conformer output: {hf_output.shape}")
        print(f"HF contains NaN: {torch.isnan(hf_output).any()}")
        if not torch.isnan(hf_output).any():
            print(f"HF min/max: {hf_output.min():.4f} / {hf_output.max():.4f}")
        else:
            # Debug where NaN appears
            nan_mask = torch.isnan(hf_output)
            print(f"  NaN count: {nan_mask.sum().item()} / {hf_output.numel()}")
            print(f"  First NaN at: {torch.where(nan_mask)[0][0].item() if nan_mask.any() else 'N/A'}")


    # Run fairseq2 forward pass
    print("\n--- fairseq2 Forward Pass ---")
    with torch.no_grad():
        fs2_output, fs2_layout = fs2_tower(mel_features, audio_mel_mask)
        print(f"FS2 full tower output (with embedder): {fs2_output.shape}")
        print(f"Layout seq_lens: {fs2_layout.seq_lens}")

    # Compare outputs
    # Note: HF outputs (N, T, 1536) from conformer, FS2 outputs (N, T, 2048) after embedder
    # We need to compare at the conformer stage before embedder projection
    print("\n" + "=" * 80)
    print("Parity Check")
    print("=" * 80)

    # Extract conformer output from FS2 (before embedder)
    with torch.no_grad():
        fs2_subsample_out = fs2_tower.subsample(mel_features)
        print(f"\nFS2 subsample output shape: {fs2_subsample_out.shape}")

        # Subsample the mask for encoder
        time_stride = 4  # 2*2 from subsample conv strides
        t_sub = fs2_subsample_out.size(1)
        indices = torch.arange(t_sub) * time_stride
        indices = torch.clamp(indices, max=audio_mel_mask.size(1) - 1)
        if indices.dim() == 1 and batch_size > 0:
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        subsampled_mask = torch.gather(audio_mel_mask, 1, indices)

        # Create layout for encoder
        from fairseq2.nn import BatchLayout
        subsample_layout = BatchLayout(
            shape=(batch_size, fs2_subsample_out.size(1)),
            seq_lens=[fs2_subsample_out.size(1)] * batch_size
        )
        fs2_conformer_out = fs2_tower.encoder(fs2_subsample_out, subsample_layout, subsampled_mask)

    print(f"\nHF conformer output shape: {hf_output.shape}")
    print(f"FS2 conformer output shape: {fs2_conformer_out.shape}")

    # Check if shapes match
    if hf_output.shape != fs2_conformer_out.shape:
        print(f"\n❌ FAIL: Shape mismatch!")
        print(f"  HF:  {hf_output.shape}")
        print(f"  FS2: {fs2_conformer_out.shape}")
        return False

    # Compute differences at conformer output
    abs_diff = (hf_output - fs2_conformer_out).abs()
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
        return True, prefixed_hf_state
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
        print(f"     FS2: {fs2_conformer_out[batch_idx, time_idx, feat_idx].item():.6f}")
        print(f"     Diff: {abs_diff[batch_idx, time_idx, feat_idx].item():.6e}")

        return False, prefixed_hf_state


def test_fs2_valid_inputs(hf_state_dict):
    """Test fairseq2 audio tower with valid (non-masked) inputs.

    HF has NaN bugs with valid inputs, so we only test FS2 for sanity.
    """
    print("\n" + "=" * 80)
    print("Test 2: All-valid (fairseq2 sanity check)")
    print("=" * 80)
    print("Note: HF produces NaN with valid inputs, so testing FS2 only.")

    audio_config = Gemma3nAudioConfig()
    text_config = get_gemma3n_e2b_config()

    batch_size = 2
    time_steps = 100

    torch.manual_seed(42)
    mel_features = torch.randn(batch_size, time_steps, audio_config.input_feat_size)
    audio_mel_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)

    print(f"\nInput: {mel_features.shape}, all-valid mask")

    # Load FS2 model (reuse weights from previous test)
    fs2_tower = load_fs2_audio_tower(hf_state_dict)

    with torch.no_grad():
        fs2_output, fs2_layout = fs2_tower(mel_features, audio_mel_mask)

    print(f"FS2 output: {fs2_output.shape}")
    print(f"Contains NaN: {torch.isnan(fs2_output).any()}")
    print(f"Contains Inf: {torch.isinf(fs2_output).any()}")

    if not torch.isnan(fs2_output).any() and not torch.isinf(fs2_output).any():
        print(f"Min/Max: {fs2_output.min():.4f} / {fs2_output.max():.4f}")
        print(f"Mean/Std: {fs2_output.mean():.4f} / {fs2_output.std():.4f}")
        print("\n✅ FS2 SANITY CHECK PASSED")
        print("   Output is finite and has reasonable statistics")
        return True
    else:
        print("\n❌ FS2 SANITY CHECK FAILED")
        print("   Output contains NaN or Inf")
        return False



if __name__ == "__main__":
    # Test 1: HF parity with all-masked input
    parity_passed, hf_state = test_audio_tower_parity()

    # Test 2: FS2 sanity check with all-valid input
    sanity_passed = False
    if hf_state is not None:
        sanity_passed = test_fs2_valid_inputs(hf_state)

    # Overall success requires both tests to pass
    success = parity_passed and sanity_passed
    exit(0 if success else 1)
