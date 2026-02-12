#!/usr/bin/env python3
"""Inspect audio tower keys in Gemma3n checkpoint."""

from __future__ import annotations

from transformers import AutoModel


def inspect_audio_keys() -> None:
    print("=" * 80)
    print("GEMMA3N AUDIO TOWER CHECKPOINT INSPECTION")
    print("=" * 80)

    # Load E2B instruct model (has multimodal)
    print("\nLoading google/gemma-3n-E2B-it...")
    model = AutoModel.from_pretrained("google/gemma-3n-E2B-it", trust_remote_code=True)

    state_dict = model.state_dict()

    # Find all audio-related keys
    audio_keys = [k for k in state_dict.keys() if "audio" in k.lower()]

    print(f"\nFound {len(audio_keys)} audio-related keys\n")

    # Group by prefix
    prefixes = {}
    for key in audio_keys:
        prefix = key.split(".")[0:2]
        prefix_str = ".".join(prefix)
        if prefix_str not in prefixes:
            prefixes[prefix_str] = []
        prefixes[prefix_str].append(key)

    for prefix, keys in sorted(prefixes.items()):
        print(f"\n{prefix}: ({len(keys)} keys)")
        for key in sorted(keys)[:10]:  # Show first 10
            shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], "shape") else "scalar"
            print(f"  {key}: {shape}")
        if len(keys) > 10:
            print(f"  ... and {len(keys) - 10} more")

    # Show audio_tower structure specifically
    audio_tower_keys = [k for k in state_dict.keys() if "audio_tower" in k]
    print(f"\n\nAUDIO TOWER STRUCTURE ({len(audio_tower_keys)} keys)")
    print("=" * 80)

    # Find unique layer types
    layer_types = set()
    for key in audio_tower_keys:
        # Remove any model. prefix if present
        clean_key = key.replace("model.", "")
        if clean_key.startswith("audio_tower."):
            parts = clean_key.replace("audio_tower.", "").split(".")
            if len(parts) > 0:
                layer_types.add(parts[0])

    print(f"\nTop-level components: {sorted(layer_types)}")

    # Check config
    print(f"\n\nAUDIO CONFIG")
    print("=" * 80)
    if hasattr(model.config, "audio_config"):
        audio_config = model.config.audio_config
        print(f"  hidden_size: {audio_config.hidden_size}")
        print(f"  vocab_size: {audio_config.vocab_size}")
        print(f"  vocab_offset: {audio_config.vocab_offset}")
        print(f"  input_feat_size: {audio_config.input_feat_size}")
        print(f"  rms_norm_eps: {audio_config.rms_norm_eps}")
    else:
        print("  No audio_config found")

    # Check embed_audio structure
    embed_audio_keys = [k for k in state_dict.keys() if k.startswith("model.embed_audio.")]
    print(f"\n\nEMBED_AUDIO STRUCTURE ({len(embed_audio_keys)} keys)")
    print("=" * 80)
    for key in sorted(embed_audio_keys):
        shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], "shape") else "scalar"
        print(f"  {key}: {shape}")


if __name__ == "__main__":
    inspect_audio_keys()
