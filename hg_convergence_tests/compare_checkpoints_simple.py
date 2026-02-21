#!/usr/bin/env python3
"""
Simple Checkpoint Comparison Script

Compares two PyTorch checkpoints saved in the same format.
Both fairseq2 and HuggingFace now save checkpoints as:
  checkpoint_500.pt = {"model": state_dict}

Usage:
    python scripts/compare_checkpoints.py \
        --fairseq2-checkpoint out/fairseq2/checkpoint_500.pt \
        --hf-checkpoint out/hf/checkpoint_500.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: Path) -> dict:
    """Load a PyTorch checkpoint."""
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")

    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint missing 'model' key: {path}")

    state_dict = checkpoint["model"]
    print(f"  Loaded {len(state_dict)} parameters")

    return state_dict


def compare_state_dicts(
    state1: dict,
    state2: dict,
    name1: str = "checkpoint1",
    name2: str = "checkpoint2",
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> tuple[bool, list[dict]]:
    """Compare two state dictionaries.

    Returns:
        (all_match, mismatches)
    """
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())

    # Check for key differences
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    if only_in_1:
        print(f"\n⚠ {len(only_in_1)} parameters only in {name1}:")
        for key in sorted(only_in_1)[:5]:
            print(f"  - {key}")
        if len(only_in_1) > 5:
            print(f"  ... and {len(only_in_1) - 5} more")

    if only_in_2:
        print(f"\n⚠ {len(only_in_2)} parameters only in {name2}:")
        for key in sorted(only_in_2)[:5]:
            print(f"  - {key}")
        if len(only_in_2) > 5:
            print(f"  ... and {len(only_in_2) - 5} more")

    # Compare common parameters
    common_keys = keys1 & keys2
    print(f"\n📊 Comparing {len(common_keys)} common parameters...")

    all_match = True
    mismatches = []

    for key in sorted(common_keys):
        param1 = state1[key]
        param2 = state2[key]

        # Convert to float32 for comparison
        tensor1 = param1.detach().cpu()
        tensor2 = param2.detach().cpu()

        if tensor1.dtype == torch.bfloat16:
            tensor1 = tensor1.to(torch.float32)
        if tensor2.dtype == torch.bfloat16:
            tensor2 = tensor2.to(torch.float32)

        # Shape check
        if tensor1.shape != tensor2.shape:
            all_match = False
            mismatches.append({
                "name": key,
                "error": "shape_mismatch",
                "shape1": tensor1.shape,
                "shape2": tensor2.shape,
            })
            continue

        # Value check
        np1 = tensor1.numpy()
        np2 = tensor2.numpy()

        if not np.allclose(np1, np2, rtol=rtol, atol=atol):
            all_match = False

            abs_diff = np.abs(np1 - np2)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)

            # Relative difference
            np2_abs = np.abs(np2)
            mask = np2_abs > 1e-10
            if np.any(mask):
                rel_diff = np.max(abs_diff[mask] / np2_abs[mask])
            else:
                rel_diff = 0.0 if max_diff < atol else float('inf')

            mismatches.append({
                "name": key,
                "error": "value_mismatch",
                "shape": tensor1.shape,
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "rel_diff": float(rel_diff),
            })

    return all_match, mismatches


def report_results(all_match: bool, mismatches: list[dict], top_n: int = 10) -> None:
    """Print comparison results."""
    print("\n" + "=" * 80)

    if all_match:
        print("✅ SUCCESS: All parameters match within tolerance!")
        print("=" * 80)
        return

    print(f"❌ Found {len(mismatches)} mismatches")
    print("=" * 80)

    # Shape mismatches
    shape_mismatches = [m for m in mismatches if m["error"] == "shape_mismatch"]
    if shape_mismatches:
        print(f"\n🔸 Shape mismatches ({len(shape_mismatches)}):")
        for m in shape_mismatches[:top_n]:
            print(f"  {m['name']}")
            print(f"    Shape 1: {m['shape1']}")
            print(f"    Shape 2: {m['shape2']}")

    # Value mismatches
    value_mismatches = [m for m in mismatches if m["error"] == "value_mismatch"]
    if value_mismatches:
        print(f"\n🔸 Value mismatches ({len(value_mismatches)}):")

        # Sort by max diff
        sorted_mismatches = sorted(
            value_mismatches,
            key=lambda m: m["max_diff"],
            reverse=True
        )

        print(f"\nTop {min(top_n, len(sorted_mismatches))} worst mismatches:")
        for i, m in enumerate(sorted_mismatches[:top_n], 1):
            print(f"\n{i}. {m['name']}")
            print(f"   Shape:     {m['shape']}")
            print(f"   Max diff:  {m['max_diff']:.6e}")
            print(f"   Mean diff: {m['mean_diff']:.6e}")
            print(f"   Rel diff:  {m['rel_diff']:.6e}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare two PyTorch checkpoints")
    parser.add_argument(
        "--fairseq2-checkpoint",
        type=Path,
        required=True,
        help="Path to fairseq2 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=Path,
        required=True,
        help="Path to HuggingFace checkpoint (.pt file)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance (default: 1e-5)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst mismatches to show (default: 10)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.fairseq2_checkpoint.exists():
        print(f"❌ Error: fairseq2 checkpoint not found: {args.fairseq2_checkpoint}")
        return 1

    if not args.hf_checkpoint.exists():
        print(f"❌ Error: HuggingFace checkpoint not found: {args.hf_checkpoint}")
        return 1

    print("=" * 80)
    print("🔍 Checkpoint Comparison")
    print("=" * 80)
    print(f"fairseq2:     {args.fairseq2_checkpoint}")
    print(f"HuggingFace:  {args.hf_checkpoint}")
    print(f"Tolerances:   rtol={args.rtol}, atol={args.atol}")
    print("=" * 80)

    # Load checkpoints
    fs2_state = load_checkpoint(args.fairseq2_checkpoint)
    hf_state = load_checkpoint(args.hf_checkpoint)

    # Compare
    all_match, mismatches = compare_state_dicts(
        fs2_state,
        hf_state,
        name1="fairseq2",
        name2="HuggingFace",
        rtol=args.rtol,
        atol=args.atol,
    )

    # Report
    report_results(all_match, mismatches, top_n=args.top_n)

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
