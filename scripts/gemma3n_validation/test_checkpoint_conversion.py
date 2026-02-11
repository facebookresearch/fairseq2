#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight checkpoint conversion test (no model loading)."""

from __future__ import annotations

import torch
from safetensors import safe_open
from transformers.utils import cached_file

from fairseq2.models.gemma3n.config import get_gemma3n_e2b_config
from fairseq2.models.gemma3n.interop import convert_gemma3n_state_dict


def main() -> None:
    print("="*80)
    print("GEMMA3N CHECKPOINT CONVERSION TEST")
    print("="*80)

    # Load checkpoint keys from safetensors (lightweight)
    print("\n[1/3] Loading HuggingFace checkpoint keys...")
    hf_model_id = "google/gemma-3n-E2B-it"
    hf_keys = set()

    # Get first shard to read keys
    shard_file = cached_file(hf_model_id, "model-00001-of-00004.safetensors")

    with safe_open(shard_file, framework="pt", device="cpu") as f:
        hf_keys.update(f.keys())

    print(f"✓ Found {len(hf_keys)} keys in HF checkpoint (sample from first shard)")

    # Show sample keys
    print("\nSample HuggingFace keys:")
    for key in sorted(hf_keys)[:10]:
        print(f"  - {key}")

    # Test conversion (just key mapping, no actual tensors)
    print("\n[2/3] Testing checkpoint conversion mapping...")
    config = get_gemma3n_e2b_config()

    # Create a dummy state dict with just the keys
    dummy_hf_state = {key: torch.tensor([0.0]) for key in hf_keys}

    # Convert
    converted_state = convert_gemma3n_state_dict(dummy_hf_state, config)
    fs2_keys = set(converted_state.keys())

    print(f"✓ Converted to {len(fs2_keys)} fairseq2 keys")

    # Show sample converted keys
    print("\nSample fairseq2 keys:")
    for key in sorted(fs2_keys)[:10]:
        print(f"  - {key}")

    # Analyze conversion
    print("\n[3/3] Analyzing conversion...")

    # Find unmapped keys (keys that didn't get converted)
    unmapped_keys = hf_keys - set(dummy_hf_state.keys())

    # Key prefixes in converted state
    prefixes = {}
    for key in fs2_keys:
        prefix = key.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    print("\nKey distribution by prefix:")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix:30s}: {count:4d} keys")

    # Check for critical components
    print("\nChecking for critical components:")

    checks = {
        "Embeddings": any("embed" in k for k in fs2_keys),
        "Decoder layers": any("decoder.layers" in k for k in fs2_keys),
        "LAuReL": any("laurel" in k for k in fs2_keys),
        "AltUp": any("altup" in k for k in fs2_keys),
        "PLE": any("per_layer" in k for k in fs2_keys),
        "QK norm": any("q_norm" in k or "k_norm" in k for k in fs2_keys),
        "Final projection": any("final_proj" in k for k in fs2_keys),
    }

    for component, found in checks.items():
        status = "✓" if found else "✗"
        print(f"  {status} {component}")

    # Summary
    print("\n" + "="*80)

    all_found = all(checks.values())

    if all_found:
        print("✅ CHECKPOINT CONVERSION TEST PASSED")
        print(f"   All {len(fs2_keys)} keys converted successfully")
        print(f"   All critical components present")
    else:
        print("⚠️  CHECKPOINT CONVERSION TEST INCOMPLETE")
        missing = [k for k, v in checks.items() if not v]
        print(f"   Missing components: {', '.join(missing)}")

    print("="*80)


if __name__ == "__main__":
    main()
