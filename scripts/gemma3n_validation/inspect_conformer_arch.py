#!/usr/bin/env python3
"""Detailed conformer architecture inspection."""

from __future__ import annotations

from transformers import AutoModel


def inspect_conformer_architecture() -> None:
    print("=" * 80)
    print("CONFORMER ARCHITECTURE ANALYSIS")
    print("=" * 80)

    print("\nLoading model from cache...")
    model = AutoModel.from_pretrained(
        "google/gemma-3n-E2B-it",
        trust_remote_code=True,
        local_files_only=True
    )
    state_dict = model.state_dict()

    # Count conformer layers (handle both with and without model. prefix)
    conformer_keys = [k for k in state_dict.keys() if "conformer" in k and "audio_tower" in k]

    layer_indices = set()
    for key in conformer_keys:
        clean_key = key.replace("model.", "")
        if "audio_tower.conformer." in clean_key:
            parts = clean_key.split(".")
            # Find the index after 'conformer'
            try:
                conf_idx = parts.index("conformer")
                if conf_idx + 1 < len(parts) and parts[conf_idx + 1].isdigit():
                    layer_indices.add(int(parts[conf_idx + 1]))
            except (ValueError, IndexError):
                continue

    num_layers = len(layer_indices)
    print(f"\n✓ Found {num_layers} conformer layers: {sorted(layer_indices)}")

    # Analyze single layer structure
    print("\n" + "=" * 80)
    print("LAYER 0 DETAILED STRUCTURE")
    print("=" * 80)

    # Find the first layer index
    if not layer_indices:
        print("No layers found!")
        return

    first_layer_idx = min(layer_indices)
    layer_0_keys = [k for k in conformer_keys if f"conformer.{first_layer_idx}." in k]

    components = {}
    for key in layer_0_keys:
        clean_key = key.replace("model.", "")
        parts = clean_key.replace(f"audio_tower.conformer.{first_layer_idx}.", "").split(".")
        component = parts[0]
        if component not in components:
            components[component] = []
        components[component].append(key)

    for component in sorted(components.keys()):
        print(f"\n{component}: ({len(components[component])} params)")
        for key in sorted(components[component]):
            param = state_dict[key]
            shape = tuple(param.shape) if hasattr(param, "shape") else "scalar"
            clean_key = key.replace("model.", "").replace(f"audio_tower.conformer.{first_layer_idx}.", "")
            print(f"  {clean_key}: {shape}")

    # Check if all layers have same structure
    print("\n" + "=" * 80)
    print("LAYER STRUCTURE CONSISTENCY CHECK")
    print("=" * 80)

    layer_structures = {}
    for idx in sorted(layer_indices):
        layer_keys = [k for k in conformer_keys if f"conformer.{idx}." in k]
        layer_components = set()
        for k in layer_keys:
            clean_key = k.replace("model.", "")
            parts = clean_key.replace(f"audio_tower.conformer.{idx}.", "").split(".")
            if parts:
                layer_components.add(parts[0])
        layer_structures[idx] = layer_components

    # Check if all same
    if not layer_structures:
        print("\n⚠ No layer structures found")
        return

    first_idx = min(layer_structures.keys())
    first_structure = layer_structures[first_idx]
    all_same = all(struct == first_structure for struct in layer_structures.values())

    if all_same:
        print(f"\n✓ All {num_layers} layers have identical structure")
        print(f"  Components: {sorted(first_structure)}")
    else:
        print(f"\n⚠ Layers have different structures:")
        for idx in sorted(layer_indices):
            diff = layer_structures[idx] - first_structure
            missing = first_structure - layer_structures[idx]
            if diff or missing:
                print(f"  Layer {idx}: +{diff} -{missing}")

    # Analyze dimensions
    print("\n" + "=" * 80)
    print("KEY DIMENSIONS")
    print("=" * 80)

    # Find example weight keys (handle flexible prefix)
    def find_key(pattern: str) -> str | None:
        for k in state_dict.keys():
            clean_k = k.replace("model.", "")
            if pattern in clean_k:
                return k
        return None

    example_keys = {
        "Q/K/V projection": find_key(f"audio_tower.conformer.{first_layer_idx}.attention.attn.q_proj.weight"),
        "Attention output": find_key(f"audio_tower.conformer.{first_layer_idx}.attention.post.weight"),
        "FFN layer 1": find_key(f"audio_tower.conformer.{first_layer_idx}.ffw_layer_end.ffw_layer_1.weight"),
        "FFN layer 2": find_key(f"audio_tower.conformer.{first_layer_idx}.ffw_layer_end.ffw_layer_2.weight"),
    }

    for name, key in example_keys.items():
        if key:
            print(f"  {name}: {tuple(state_dict[key].shape)}")
        else:
            print(f"  {name}: NOT FOUND")

    # Check for convolution components
    print("\n" + "=" * 80)
    print("CONVOLUTION MODULE (if present)")
    print("=" * 80)

    conv_keys = [k for k in layer_0_keys if "conv" in k.lower() and "subsample" not in k.lower()]
    if conv_keys:
        for key in sorted(conv_keys):
            shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], "shape") else "scalar"
            clean_key = key.replace("model.", "").replace(f"audio_tower.conformer.{first_layer_idx}.", "")
            print(f"  {clean_key}: {shape}")
    else:
        print("  No convolution module found in conformer layers")

    # Check audio config details
    print("\n" + "=" * 80)
    print("AUDIO CONFIG FULL DETAILS")
    print("=" * 80)

    audio_config = model.config.audio_config
    for attr in dir(audio_config):
        if not attr.startswith("_") and not callable(getattr(audio_config, attr)):
            val = getattr(audio_config, attr)
            if not isinstance(val, (dict, list)) or len(str(val)) < 100:
                print(f"  {attr}: {val}")


if __name__ == "__main__":
    inspect_conformer_architecture()
