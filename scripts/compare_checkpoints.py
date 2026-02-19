#!/usr/bin/env python3
"""
Checkpoint Comparison Script for Convergence Validation

This script compares fairseq2 and Unsloth checkpoints to validate that the fairseq2
HuggingFace adapter produces identical convergence to Unsloth.

Usage:
    python scripts/compare_checkpoints.py \
        --fairseq2-checkpoint out/fairseq2/checkpoint_100.pt \
        --unsloth-checkpoint out/unsloth/checkpoint_100.pt

Exit codes:
    0: All parameters match within tolerance
    1: Some parameters differ beyond tolerance
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: Path) -> dict:
    """Load a PyTorch checkpoint file.

    Args:
        path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    print(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def extract_model_state_dict(checkpoint: dict, checkpoint_type: str) -> dict:
    """Extract model state dict from checkpoint.

    Args:
        checkpoint: Loaded checkpoint
        checkpoint_type: Either 'fairseq2' or 'unsloth'

    Returns:
        Model state dictionary
    """
    if checkpoint_type == "fairseq2":
        # fairseq2 checkpoint structure: checkpoint['model']['model']
        model_state = checkpoint.get("model", {}).get("model", {})
        if not model_state:
            raise ValueError("Could not find model state in fairseq2 checkpoint")
        return model_state
    elif checkpoint_type == "unsloth":
        # Unsloth checkpoint structure: checkpoint['model_state_dict']
        model_state = checkpoint.get("model_state_dict", {})
        if not model_state:
            raise ValueError("Could not find model_state_dict in Unsloth checkpoint")
        return model_state
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")


def map_fairseq2_to_unsloth_names(fairseq2_state: dict) -> dict:
    """Map fairseq2 parameter names to Unsloth parameter names.

    fairseq2 wraps HuggingFace models with a '_wrapped_hf_model.' prefix.
    This function strips that prefix to match Unsloth naming.

    Args:
        fairseq2_state: fairseq2 model state dictionary

    Returns:
        Dictionary with Unsloth-compatible parameter names
    """
    prefix = "_wrapped_hf_model."
    mapped_state = {}

    for name, param in fairseq2_state.items():
        if name.startswith(prefix):
            unsloth_name = name[len(prefix):]
            mapped_state[unsloth_name] = param
        else:
            # Keep parameters without the prefix as-is
            mapped_state[name] = param

    print(f"Mapped {len(mapped_state)} parameters from fairseq2 to Unsloth naming")
    return mapped_state


def compare_parameters(
    fairseq2_state: dict,
    unsloth_state: dict,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> tuple[bool, list[dict]]:
    """Compare parameters between fairseq2 and Unsloth checkpoints.

    Args:
        fairseq2_state: fairseq2 model state (with mapped names)
        unsloth_state: Unsloth model state
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Returns:
        Tuple of (all_match, mismatches) where:
            - all_match: True if all parameters match within tolerance
            - mismatches: List of dicts with mismatch information
    """
    fairseq2_keys = set(fairseq2_state.keys())
    unsloth_keys = set(unsloth_state.keys())

    # Check for missing parameters
    only_in_fairseq2 = fairseq2_keys - unsloth_keys
    only_in_unsloth = unsloth_keys - fairseq2_keys

    if only_in_fairseq2:
        print(f"WARNING: {len(only_in_fairseq2)} parameters only in fairseq2:")
        for name in sorted(only_in_fairseq2)[:10]:  # Show first 10
            print(f"  - {name}")
        if len(only_in_fairseq2) > 10:
            print(f"  ... and {len(only_in_fairseq2) - 10} more")

    if only_in_unsloth:
        print(f"WARNING: {len(only_in_unsloth)} parameters only in Unsloth:")
        for name in sorted(only_in_unsloth)[:10]:  # Show first 10
            print(f"  - {name}")
        if len(only_in_unsloth) > 10:
            print(f"  ... and {len(only_in_unsloth) - 10} more")

    # Compare common parameters
    common_keys = fairseq2_keys & unsloth_keys
    print(f"\nComparing {len(common_keys)} common parameters...")

    all_match = True
    mismatches = []

    for name in sorted(common_keys):
        fairseq2_param = fairseq2_state[name]
        unsloth_param = unsloth_state[name]

        # Convert to numpy for comparison
        fairseq2_np = fairseq2_param.detach().cpu().numpy()
        unsloth_np = unsloth_param.detach().cpu().numpy()

        # Check shape match
        if fairseq2_np.shape != unsloth_np.shape:
            all_match = False
            mismatches.append({
                "name": name,
                "error": "shape_mismatch",
                "fairseq2_shape": fairseq2_np.shape,
                "unsloth_shape": unsloth_np.shape,
                "max_diff": None,
                "mean_diff": None,
                "rel_diff": None,
            })
            continue

        # Check value match
        if not np.allclose(fairseq2_np, unsloth_np, rtol=rtol, atol=atol):
            all_match = False

            # Calculate detailed statistics
            abs_diff = np.abs(fairseq2_np - unsloth_np)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)

            # Relative difference (avoid division by zero)
            unsloth_abs = np.abs(unsloth_np)
            mask = unsloth_abs > 1e-10
            if np.any(mask):
                rel_diff = np.max(abs_diff[mask] / unsloth_abs[mask])
            else:
                rel_diff = 0.0 if max_diff < atol else float('inf')

            mismatches.append({
                "name": name,
                "error": "value_mismatch",
                "fairseq2_shape": fairseq2_np.shape,
                "unsloth_shape": unsloth_np.shape,
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "rel_diff": float(rel_diff),
            })

    return all_match, mismatches


def report_mismatches(mismatches: list[dict], top_n: int = 10) -> None:
    """Report detailed mismatch information.

    Args:
        mismatches: List of mismatch dictionaries
        top_n: Number of worst mismatches to show in detail
    """
    if not mismatches:
        print("\n✓ All parameters match within tolerance!")
        return

    print(f"\n✗ Found {len(mismatches)} mismatches:")

    # Separate shape and value mismatches
    shape_mismatches = [m for m in mismatches if m["error"] == "shape_mismatch"]
    value_mismatches = [m for m in mismatches if m["error"] == "value_mismatch"]

    if shape_mismatches:
        print(f"\nShape mismatches ({len(shape_mismatches)}):")
        for m in shape_mismatches[:top_n]:
            print(f"  - {m['name']}")
            print(f"    fairseq2: {m['fairseq2_shape']}, Unsloth: {m['unsloth_shape']}")

    if value_mismatches:
        print(f"\nValue mismatches ({len(value_mismatches)}):")

        # Sort by max absolute difference
        value_mismatches_sorted = sorted(
            value_mismatches,
            key=lambda m: m["max_diff"],
            reverse=True
        )

        print(f"\nTop {min(top_n, len(value_mismatches))} worst mismatches by absolute difference:")
        for i, m in enumerate(value_mismatches_sorted[:top_n], 1):
            print(f"\n{i}. {m['name']}")
            print(f"   Shape: {m['fairseq2_shape']}")
            print(f"   Max diff:  {m['max_diff']:.6e}")
            print(f"   Mean diff: {m['mean_diff']:.6e}")
            print(f"   Rel diff:  {m['rel_diff']:.6e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare fairseq2 and Unsloth checkpoints for convergence validation"
    )
    parser.add_argument(
        "--fairseq2-checkpoint",
        type=Path,
        required=True,
        help="Path to fairseq2 checkpoint file (checkpoint_100.pt)",
    )
    parser.add_argument(
        "--unsloth-checkpoint",
        type=Path,
        required=True,
        help="Path to Unsloth checkpoint file (checkpoint_100.pt)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for np.allclose (default: 1e-3, appropriate for bf16)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for np.allclose (default: 1e-5, appropriate for bf16)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst mismatches to report in detail (default: 10)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.fairseq2_checkpoint.exists():
        print(f"ERROR: fairseq2 checkpoint not found: {args.fairseq2_checkpoint}")
        return 1

    if not args.unsloth_checkpoint.exists():
        print(f"ERROR: Unsloth checkpoint not found: {args.unsloth_checkpoint}")
        return 1

    print("=" * 80)
    print("Checkpoint Comparison for Convergence Validation")
    print("=" * 80)
    print(f"fairseq2 checkpoint: {args.fairseq2_checkpoint}")
    print(f"Unsloth checkpoint:  {args.unsloth_checkpoint}")
    print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
    print("=" * 80)

    # Load checkpoints
    fairseq2_checkpoint = load_checkpoint(args.fairseq2_checkpoint)
    unsloth_checkpoint = load_checkpoint(args.unsloth_checkpoint)

    # Extract model state dicts
    print("\nExtracting model state dictionaries...")
    fairseq2_state = extract_model_state_dict(fairseq2_checkpoint, "fairseq2")
    unsloth_state = extract_model_state_dict(unsloth_checkpoint, "unsloth")

    print(f"fairseq2 state: {len(fairseq2_state)} parameters")
    print(f"Unsloth state:  {len(unsloth_state)} parameters")

    # Map fairseq2 names to Unsloth names
    print("\nMapping parameter names...")
    fairseq2_state_mapped = map_fairseq2_to_unsloth_names(fairseq2_state)

    # Compare parameters
    print("\n" + "=" * 80)
    all_match, mismatches = compare_parameters(
        fairseq2_state_mapped,
        unsloth_state,
        rtol=args.rtol,
        atol=args.atol
    )

    # Report results
    print("=" * 80)
    report_mismatches(mismatches, top_n=args.top_n)
    print("=" * 80)

    if all_match:
        print("\n✓ SUCCESS: All parameters match within tolerance")
        return 0
    else:
        print(f"\n✗ FAILURE: {len(mismatches)} parameters differ beyond tolerance")
        return 1


if __name__ == "__main__":
    sys.exit(main())
