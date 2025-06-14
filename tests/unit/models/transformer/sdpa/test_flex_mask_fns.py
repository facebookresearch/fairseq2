import torch
from unittest.mock import patch

from fairseq2.models.transformer._sdpa._flex import (
    _causal_mask_fn,
    _sliding_window_causal_mask_fn,
    _offsets_to_doc_ids_tensor,
    _create_packed_mask_fn,
    _create_padding_mask_fn,
    _dropout_mask_fn,
)


def test_causal_mask():
    b, h = torch.tensor([0]), torch.tensor([0])

    # q_idx >= kv_idx should be True
    assert _causal_mask_fn(b, h, torch.tensor([2]), torch.tensor([1])).item()
    assert _causal_mask_fn(b, h, torch.tensor([2]), torch.tensor([2])).item()

    # q_idx < kv_idx should be False
    assert not _causal_mask_fn(b, h, torch.tensor([1]), torch.tensor([2])).item()


def test_sliding_window_mask():
    mask_fn = _sliding_window_causal_mask_fn(window_size=2)
    b, h = torch.tensor([0]), torch.tensor([0])

    # Within window and causal
    assert mask_fn(b, h, torch.tensor([3]), torch.tensor([2])).item()  # distance=1
    assert mask_fn(b, h, torch.tensor([3]), torch.tensor([1])).item()  # distance=2

    # Outside window
    assert not mask_fn(b, h, torch.tensor([4]), torch.tensor([1])).item()  # distance=3

    # Non-causal
    assert not mask_fn(b, h, torch.tensor([1]), torch.tensor([2])).item()


def test_offsets_to_doc_ids():
    offsets = torch.tensor([0, 3, 5])  # Two docs: [0,1,2] and [3,4]
    doc_ids = _offsets_to_doc_ids_tensor(offsets)
    expected = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    assert torch.equal(doc_ids, expected)


def test_packed_mask():
    seq_offsets = torch.tensor([0, 3, 6])  # Two docs of length 3 each
    key_offsets = torch.tensor([0, 3, 6])
    mask_fn = _create_packed_mask_fn(seq_offsets, key_offsets)

    b, h = torch.tensor([0]), torch.tensor([0])

    # Same document
    assert mask_fn(b, h, torch.tensor([1]), torch.tensor([0])).item()

    # Different documents
    assert not mask_fn(b, h, torch.tensor([4]), torch.tensor([1])).item()


def test_packed_mask_with_base_mask():
    seq_offsets = torch.tensor([0, 3, 6])
    key_offsets = torch.tensor([0, 3, 6])
    mask_fn = _create_packed_mask_fn(seq_offsets, key_offsets, _causal_mask_fn)

    b, h = torch.tensor([0]), torch.tensor([0])

    # Same doc, causal
    assert mask_fn(b, h, torch.tensor([2]), torch.tensor([1])).item()

    # Same doc, non-causal
    assert not mask_fn(b, h, torch.tensor([1]), torch.tensor([2])).item()


def test_padding_mask():
    seq_lens = torch.tensor([3, 4])
    value_seq_lens = torch.tensor([2, 4])
    mask_fn = _create_padding_mask_fn(seq_lens, value_seq_lens)

    h = torch.tensor([0])

    # Valid positions
    assert mask_fn(torch.tensor([0]), h, torch.tensor([2]), torch.tensor([1])).item()

    # Invalid query position
    assert not mask_fn(
        torch.tensor([0]), h, torch.tensor([3]), torch.tensor([1])
    ).item()

    # Invalid key position
    assert not mask_fn(
        torch.tensor([0]), h, torch.tensor([1]), torch.tensor([2])
    ).item()


def test_dropout_mask():
    # No mask when not training or dropout_p=0
    assert _dropout_mask_fn(0.5, training=False) is None
    assert _dropout_mask_fn(0.0, training=True) is None

    # Returns callable when training=True and dropout_p > 0
    mask_fn = _dropout_mask_fn(0.5, training=True)
    assert callable(mask_fn)


@patch("torch.rand")
def test_dropout_mask_behavior(mock_rand):
    mask_fn = _dropout_mask_fn(0.3, training=True)
    b, h, q, kv = (
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
    )

    # Below threshold - should mask (False)
    mock_rand.return_value = torch.tensor([0.2])
    assert not mask_fn(b, h, q, kv).item()

    # Above threshold - should not mask (True)
    mock_rand.return_value = torch.tensor([0.5])
    assert mask_fn(b, h, q, kv).item()


def test_causal_mask_large_indices():
    b, h = torch.tensor([0]), torch.tensor([0])

    # Test with larger indices
    q_indices = torch.tensor([100, 50, 75])
    kv_indices = torch.tensor([50, 100, 75])

    result = _causal_mask_fn(b.expand(3), h.expand(3), q_indices, kv_indices)
    expected = torch.tensor([True, False, True])  # [100>=50, 50>=100, 75>=75]
    assert torch.equal(result, expected)


def test_packed_mask_unequal_offsets():
    # Different sequence and key lengths
    seq_offsets = torch.tensor([0, 3, 5])  # [doc0:3, doc1:2]
    key_offsets = torch.tensor([0, 2, 6])  # [doc0:2, doc1:4]
    mask_fn = _create_packed_mask_fn(seq_offsets, key_offsets)

    b, h = torch.tensor([0]), torch.tensor([0])

    # Same document, different lengths
    assert mask_fn(
        b, h, torch.tensor([1]), torch.tensor([0])
    ).item()  # doc0 query to doc0 key
    assert mask_fn(
        b, h, torch.tensor([3]), torch.tensor([2])
    ).item()  # doc1 query to doc1 key


def test_sliding_window_vectorized():
    mask_fn = _sliding_window_causal_mask_fn(window_size=2)
    b = torch.tensor([0, 0, 0, 0])
    h = torch.tensor([0, 0, 0, 0])
    q_idx = torch.tensor([3, 3, 3, 2])
    kv_idx = torch.tensor([1, 2, 3, 4])  # distances: [2, 1, 0, -2]

    result = mask_fn(b, h, q_idx, kv_idx)
    expected = torch.tensor(
        [True, True, True, False]
    )  # [causal+window, causal+window, causal+window, non-causal]
    assert torch.equal(result, expected)
