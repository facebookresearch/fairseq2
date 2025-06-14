#!/usr/bin/env python3
"""
Script to test materialize_block_mask_from_object with various block masks.
Tests different mask patterns and validates correctness using the existing functions.
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, and_masks

# Import the functions from the module (assuming they're available)
from fairseq2.models.transformer._sdpa._flex import (
    materialize_block_mask_from_object,
    _causal_mask_fn,
    _sliding_window_causal_mask_fn,
    _create_padding_mask_fn,
    _create_packed_mask_fn,
    _offsets_to_doc_ids_tensor,
)


def create_reference_mask_from_function(mask_fn, shape, block_size):
    """Create reference mask by applying mask function at block level"""
    B, H, Q_LEN, KV_LEN = shape
    Q_BLOCKS = (Q_LEN + block_size - 1) // block_size
    KV_BLOCKS = (KV_LEN + block_size - 1) // block_size

    mask = torch.zeros(B, H, Q_LEN, KV_LEN, dtype=torch.bool)

    for b in range(B):
        for h in range(H):
            for q_block in range(Q_BLOCKS):
                for kv_block in range(KV_BLOCKS):
                    if mask_fn(
                        torch.tensor(b),
                        torch.tensor(h),
                        torch.tensor(q_block),
                        torch.tensor(kv_block),
                    ):
                        # Fill the entire block
                        q_start = q_block * block_size
                        q_end = min(q_start + block_size, Q_LEN)
                        kv_start = kv_block * block_size
                        kv_end = min(kv_start + block_size, KV_LEN)

                        mask[b, h, q_start:q_end, kv_start:kv_end] = True

    return mask


def compare_masks(materialized_mask, reference_mask, name):
    """Compare materialized mask with reference mask"""
    print(f"\n--- Comparison for {name} ---")
    print(f"Materialized shape: {materialized_mask.shape}")
    print(f"Reference shape: {reference_mask.shape}")

    if materialized_mask.shape == reference_mask.shape:
        matches = torch.equal(materialized_mask, reference_mask)
        print(f"Masks match: {matches}")

        if not matches:
            diff = materialized_mask != reference_mask
            print(f"Number of differences: {diff.sum().item()}")
            print(f"Total elements: {materialized_mask.numel()}")
            print(
                f"Difference percentage: {100 * diff.sum().item() / materialized_mask.numel():.2f}%"
            )

        # Sparsity comparison
        mat_sparsity = materialized_mask.sum().item() / materialized_mask.numel()
        ref_sparsity = reference_mask.sum().item() / reference_mask.numel()
        print(f"Materialized sparsity: {mat_sparsity:.4f}")
        print(f"Reference sparsity: {ref_sparsity:.4f}")

        return matches
    else:
        print("Shape mismatch!")
        return False


def test_basic_masks():
    """Test basic mask functions"""
    print("=" * 80)
    print("TESTING BASIC MASK FUNCTIONS")
    print("=" * 80)

    B, H = 1, 1
    Q_LEN, KV_LEN = 128, 128
    BLOCK_SIZE = 32

    test_cases = [
        ("Causal Mask", _causal_mask_fn),
        ("Sliding Window (window=2)", _sliding_window_causal_mask_fn(2)),
        ("Sliding Window (window=5)", _sliding_window_causal_mask_fn(5)),
    ]

    for name, mask_fn in test_cases:
        print(f"\nTesting: {name}")
        print("-" * 40)

        # Create block mask
        block_mask = create_block_mask(
            mask_fn, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, BLOCK_SIZE=BLOCK_SIZE
        )

        print(f"Block mask shape: {block_mask.shape}")
        print(f"Block mask sparsity: {block_mask.sparsity():.2f}%")

        # Materialize using our function
        materialized = materialize_block_mask_from_object(block_mask)

        # Create reference
        reference = create_reference_mask_from_function(
            mask_fn, (B, H, Q_LEN, KV_LEN), BLOCK_SIZE
        )

        # Compare
        matches = compare_masks(materialized, reference, name)


def test_padding_masks():
    """Test padding mask functionality"""
    print("=" * 80)
    print("TESTING PADDING MASKS")
    print("=" * 80)

    B, H = 2, 1
    MAX_SEQ_LEN = 96
    BLOCK_SIZE = 32

    # Create different sequence lengths for each batch
    seq_lens = torch.tensor([64, 80])  # Different lengths per batch
    value_seq_lens = torch.tensor([64, 80])

    padding_mask_fn = _create_padding_mask_fn(seq_lens, value_seq_lens)

    print("Testing: Padding Mask")
    print(f"Sequence lengths: {seq_lens.tolist()}")

    # Create block mask
    block_mask = create_block_mask(
        padding_mask_fn,
        B=B,
        H=H,
        Q_LEN=MAX_SEQ_LEN,
        KV_LEN=MAX_SEQ_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    print(f"Block mask shape: {block_mask.shape}")

    # Materialize
    materialized = materialize_block_mask_from_object(block_mask)

    print(f"Materialized shape: {materialized.shape}")

    # Check that padding is correctly masked
    for b in range(B):
        seq_len = seq_lens[b].item()
        val_len = value_seq_lens[b].item()

        # Check that valid positions are potentially unmasked
        valid_region = materialized[b, 0, :seq_len, :val_len]
        invalid_q = materialized[b, 0, seq_len:, :]
        invalid_kv = materialized[b, 0, :, val_len:]

        print(
            f"Batch {b}: seq_len={seq_len}, valid_region_any={valid_region.any().item()}"
        )
        print(
            f"Batch {b}: invalid_q_any={invalid_q.any().item()}, invalid_kv_any={invalid_kv.any().item()}"
        )


def test_combined_masks():
    """Test combining multiple masks using and_masks"""
    print("=" * 80)
    print("TESTING COMBINED MASKS")
    print("=" * 80)

    B, H = 1, 1
    Q_LEN, KV_LEN = 96, 96
    BLOCK_SIZE = 32

    # Create individual masks
    causal_fn = _causal_mask_fn
    sliding_fn = _sliding_window_causal_mask_fn(3)

    # Test individual masks
    print("Testing individual masks...")

    causal_block = create_block_mask(
        causal_fn, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, BLOCK_SIZE=BLOCK_SIZE
    )
    sliding_block = create_block_mask(
        sliding_fn, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, BLOCK_SIZE=BLOCK_SIZE
    )

    causal_mat = materialize_block_mask_from_object(causal_block)
    sliding_mat = materialize_block_mask_from_object(sliding_block)

    print(f"Causal sparsity: {causal_mat.sum().item() / causal_mat.numel():.4f}")
    print(f"Sliding sparsity: {sliding_mat.sum().item() / sliding_mat.numel():.4f}")

    # Test combined mask
    print("\nTesting combined mask...")
    combined_fn = and_masks(causal_fn, sliding_fn)
    combined_block = create_block_mask(
        combined_fn, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, BLOCK_SIZE=BLOCK_SIZE
    )
    combined_mat = materialize_block_mask_from_object(combined_block)

    print(f"Combined sparsity: {combined_mat.sum().item() / combined_mat.numel():.4f}")

    # Verify that combined mask is intersection of individual masks
    expected_combined = causal_mat & sliding_mat
    matches = torch.equal(combined_mat, expected_combined)
    print(f"Combined mask matches intersection: {matches}")


def test_packed_sequences():
    """Test packed sequence masks"""
    print("=" * 80)
    print("TESTING PACKED SEQUENCES")
    print("=" * 80)

    # Create packed sequence layout
    # Simulate 3 documents of lengths [20, 30, 25]
    doc_lengths = [20, 30, 25]
    seq_begin_indices = torch.tensor(
        [0] + [sum(doc_lengths[: i + 1]) for i in range(len(doc_lengths))]
    )
    total_len = sum(doc_lengths)

    print(f"Document lengths: {doc_lengths}")
    print(f"Begin indices: {seq_begin_indices.tolist()}")
    print(f"Total length: {total_len}")

    BLOCK_SIZE = 16

    # Test packed mask without base mask
    print("\nTesting packed mask (no base mask)...")
    packed_fn = _create_packed_mask_fn(seq_begin_indices, seq_begin_indices, None)

    packed_block = create_block_mask(
        packed_fn, B=1, H=1, Q_LEN=total_len, KV_LEN=total_len, BLOCK_SIZE=BLOCK_SIZE
    )

    packed_mat = materialize_block_mask_from_object(packed_block)

    print(f"Packed mask shape: {packed_mat.shape}")
    print(f"Packed mask sparsity: {packed_mat.sum().item() / packed_mat.numel():.4f}")

    # Verify document boundaries
    doc_ids = _offsets_to_doc_ids_tensor(seq_begin_indices)
    print(f"Document IDs: {doc_ids[:10].tolist()}...{doc_ids[-10:].tolist()}")

    # Test packed mask with causal base mask
    print("\nTesting packed mask with causal base...")
    packed_causal_fn = _create_packed_mask_fn(
        seq_begin_indices, seq_begin_indices, _causal_mask_fn
    )

    packed_causal_block = create_block_mask(
        packed_causal_fn,
        B=1,
        H=1,
        Q_LEN=total_len,
        KV_LEN=total_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    packed_causal_mat = materialize_block_mask_from_object(packed_causal_block)

    print(
        f"Packed causal sparsity: {packed_causal_mat.sum().item() / packed_causal_mat.numel():.4f}"
    )


def test_edge_cases():
    """Test edge cases and different configurations"""
    print("=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)

    test_configs = [
        (
            "Small sequence",
            {"B": 1, "H": 1, "Q_LEN": 32, "KV_LEN": 32, "BLOCK_SIZE": 16},
        ),
        ("Non-square", {"B": 1, "H": 1, "Q_LEN": 64, "KV_LEN": 96, "BLOCK_SIZE": 16}),
        (
            "Multiple heads",
            {"B": 1, "H": 4, "Q_LEN": 64, "KV_LEN": 64, "BLOCK_SIZE": 16},
        ),
        (
            "Multiple batch",
            {"B": 3, "H": 2, "Q_LEN": 48, "KV_LEN": 48, "BLOCK_SIZE": 16},
        ),
        (
            "Large block size",
            {"B": 1, "H": 1, "Q_LEN": 64, "KV_LEN": 64, "BLOCK_SIZE": 64},
        ),
    ]

    for name, config in test_configs:
        print(f"\nTesting: {name}")
        print(f"Config: {config}")

        block_mask = create_block_mask(
            _causal_mask_fn,
            B=config["B"],
            H=config["H"],
            Q_LEN=config["Q_LEN"],
            KV_LEN=config["KV_LEN"],
            BLOCK_SIZE=config["BLOCK_SIZE"],
        )

        materialized = materialize_block_mask_from_object(block_mask)
        reference = create_reference_mask_from_function(
            _causal_mask_fn,
            (config["B"], config["H"], config["Q_LEN"], config["KV_LEN"]),
            config["BLOCK_SIZE"],
        )

        matches = compare_masks(materialized, reference, name)
        print(f"✓ Passed: {matches}")


def main():
    """Run all tests"""
    print("Starting Block Mask Materialization Tests")
    print("=" * 80)

    try:
        test_basic_masks()
        test_padding_masks()
        test_combined_masks()
        test_packed_sequences()
        test_edge_cases()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
