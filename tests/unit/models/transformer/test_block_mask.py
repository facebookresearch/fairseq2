from unittest.mock import Mock, patch

import pytest
import torch

from fairseq2.device import Device
from fairseq2.models.transformer._attention_bias import IdentityBias
from fairseq2.models.transformer._block_mask import (
    BlockMaskCache,
    BlockMaskCacheKey,
    _causal_mask_fn,
    _create_composed_mask,
    _create_packed_mask_fn,
    _create_padding_mask_fn,
    _offsets_to_doc_ids_tensor,
    _sliding_window_causal_mask_fn,
)


class TestMaskFunctions:
    """Test individual mask functions."""

    def test_causal_mask_fn(self) -> None:
        """Test causal mask function behavior."""
        q_lens = torch.tensor([3, 2])
        kv_lens = torch.tensor([3, 2])
        mask_fn = _causal_mask_fn(q_lens, kv_lens)

        # Test for batch 0
        b = torch.tensor(0)
        h = torch.tensor(0)

        # Test diagonal and upper triangular positions
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(0)) == True
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(0)) == True
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(1)) == True
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(1)) == False
        assert mask_fn(b, h, torch.tensor(2), torch.tensor(1)) == True

    def test_sliding_window_causal_mask_fn(self) -> None:
        """Test sliding window causal mask function."""
        q_lens = torch.tensor([4])
        kv_lens = torch.tensor([4])
        window_size = 2
        mask_fn = _sliding_window_causal_mask_fn(window_size, q_lens, kv_lens)

        b = torch.tensor(0)
        h = torch.tensor(0)

        # Test window behavior
        assert mask_fn(b, h, torch.tensor(2), torch.tensor(1)) == True  # Within window
        assert mask_fn(b, h, torch.tensor(2), torch.tensor(2)) == True  # Diagonal
        assert (
            mask_fn(b, h, torch.tensor(3), torch.tensor(1)) == False
        )  # Outside window
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(2)) == False  # Future token

    def test_sliding_window_size_one(self) -> None:
        """Test sliding window with size 1 (diagonal only)."""
        q_lens = torch.tensor([3])
        kv_lens = torch.tensor([3])
        mask_fn = _sliding_window_causal_mask_fn(1, q_lens, kv_lens)

        b = torch.tensor(0)
        h = torch.tensor(0)

        # Only diagonal should be True
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(0)) == True
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(1)) == True
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(0)) == False
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(1)) == False

    def test_offsets_to_doc_ids_tensor(self) -> None:
        """Test conversion of offsets to document IDs."""
        offsets = torch.tensor([0, 3, 5, 8])
        doc_ids = _offsets_to_doc_ids_tensor(offsets)
        expected = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        assert torch.equal(doc_ids, expected)

    def test_padding_mask_fn(self) -> None:
        """Test padding mask function."""
        q_lens = torch.tensor([2, 3])
        kv_lens = torch.tensor([3, 2])
        mask_fn = _create_padding_mask_fn(q_lens, kv_lens)

        b = torch.tensor(0)
        h = torch.tensor(0)

        # Valid positions
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(0)) == True
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(2)) == True
        # Invalid positions (beyond sequence length)
        assert mask_fn(b, h, torch.tensor(2), torch.tensor(0)) == False
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(3)) == False


class TestPackedMaskFunction:
    """Test packed sequence mask function."""

    def test_create_packed_mask_fn_basic(self) -> None:
        """Test basic packed mask functionality."""
        seq_begin_indices = torch.tensor([0, 3, 5])
        keys_begin_indices = torch.tensor([0, 3, 5])

        mask_fn = _create_packed_mask_fn(seq_begin_indices, keys_begin_indices)

        b = torch.tensor(0)
        h = torch.tensor(0)

        # Same document
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(1)) == True
        assert mask_fn(b, h, torch.tensor(3), torch.tensor(4)) == True
        # Different documents
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(3)) == False
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(4)) == False

    def test_create_packed_mask_fn_with_base_mask(self) -> None:
        """Test packed mask with base causal mask."""
        seq_begin_indices = torch.tensor([0, 2, 4])
        keys_begin_indices = torch.tensor([0, 2, 4])
        q_lens = torch.tensor([2, 2])
        kv_lens = torch.tensor([2, 2])

        base_mask_fn = _causal_mask_fn(q_lens, kv_lens)
        mask_fn = _create_packed_mask_fn(
            seq_begin_indices, keys_begin_indices, base_mask_fn
        )

        b = torch.tensor(0)
        h = torch.tensor(0)

        # Same document, causal valid
        assert mask_fn(b, h, torch.tensor(1), torch.tensor(0)) == True
        # Same document, causal invalid
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(1)) == False
        # Different documents
        assert mask_fn(b, h, torch.tensor(0), torch.tensor(2)) == False


class TestBlockMaskCache:
    """Test block mask caching functionality."""

    def test_cache_key_creation(self) -> None:
        """Test cache key creation for different layouts."""
        cache = BlockMaskCache()

        # Mock BatchLayout for non-packed sequences
        seqs_layout = Mock()
        seqs_layout.packed = False
        seqs_layout.seq_lens = [3, 4, 2]
        seqs_layout.max_seq_len = 4

        keys_layout = Mock()
        keys_layout.packed = False
        keys_layout.seq_lens = [3, 4, 2]
        keys_layout.max_seq_len = 4

        key = cache._create_cache_key(seqs_layout, keys_layout)
        assert key.batch_size == 3
        assert key.seqs_len == 4
        assert key.keys_len == 4

    def test_cache_key_creation_packed(self) -> None:
        """Test cache key creation for packed sequences."""
        cache = BlockMaskCache()

        # Mock BatchLayout for packed sequences
        seqs_layout = Mock()
        seqs_layout.packed = True
        seqs_layout.seq_begin_indices = [0, 3, 7]

        keys_layout = Mock()
        keys_layout.packed = True
        keys_layout.seq_begin_indices = [0, 3, 7]

        key = cache._create_cache_key(seqs_layout, keys_layout)
        assert key.batch_size == 1
        assert key.seqs_len == 7
        assert key.keys_len == 7

    def test_cache_key_hash(self) -> None:
        """Test that cache keys are hashable."""
        key1 = BlockMaskCacheKey(batch_size=2, seqs_len=10, keys_len=10)
        key2 = BlockMaskCacheKey(batch_size=2, seqs_len=10, keys_len=10)
        key3 = BlockMaskCacheKey(batch_size=3, seqs_len=10, keys_len=10)

        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)
        assert key1 == key2
        assert key1 != key3

    @patch("fairseq2.models.transformer._block_mask._create_composed_mask")
    def test_cache_hit_and_miss(self, mock_create_mask: Mock) -> None:
        """Test cache hit and miss behavior."""
        cache = BlockMaskCache()
        mock_mask = Mock()
        mock_create_mask.return_value = mock_mask

        # Mock inputs
        bias = Mock(spec=IdentityBias)
        seqs_layout = Mock()
        seqs_layout.packed = False
        seqs_layout.seq_lens = [3, 4]
        seqs_layout.max_seq_len = 4

        keys_layout = Mock()
        keys_layout.packed = False
        keys_layout.seq_lens = [3, 4]
        keys_layout.max_seq_len = 4

        device = Mock(spec=Device)

        # First call - cache miss
        result1 = cache.get_or_create_mask(bias, seqs_layout, keys_layout, device)
        assert result1 == mock_mask
        assert mock_create_mask.call_count == 1

        # Second call - cache hit
        result2 = cache.get_or_create_mask(bias, seqs_layout, keys_layout, device)
        assert result2 == mock_mask
        assert mock_create_mask.call_count == 1  # Should not increase


class TestCreateComposedMask:
    """Test the main composed mask creation function."""

    @patch("fairseq2.models.transformer._block_mask.create_block_mask")
    def test_create_composed_mask_identity_bias(
        self, mock_create_block_mask: Mock
    ) -> None:
        """Test composed mask creation with identity bias."""
        mock_block_mask = Mock()
        mock_create_block_mask.return_value = mock_block_mask

        bias = Mock(spec=IdentityBias)

        # Mock BatchLayout
        seqs_layout = Mock()
        seqs_layout.packed = False
        seqs_layout.padded = True
        seqs_layout.seq_lens = [3, 4]
        seqs_layout.max_seq_len = 4
        seqs_layout.seq_lens_pt = torch.tensor([3, 4])

        keys_layout = Mock()
        keys_layout.packed = False
        keys_layout.padded = True
        keys_layout.seq_lens = [3, 4]
        keys_layout.max_seq_len = 4
        keys_layout.seq_lens_pt = torch.tensor([3, 4])

        device = Mock(spec=Device)

        result = _create_composed_mask(bias, seqs_layout, keys_layout, device)

        # Should create block mask with padding mask only
        mock_create_block_mask.assert_called_once()
        assert result == mock_block_mask

    @patch("fairseq2.models.transformer._block_mask.create_block_mask")
    def test_create_composed_mask_no_masks_needed(
        self, mock_create_block_mask: Mock
    ) -> None:
        """Test when no masks are needed."""
        bias = Mock(spec=IdentityBias)

        # Mock BatchLayout with no padding
        seqs_layout = Mock()
        seqs_layout.packed = False
        seqs_layout.padded = False

        keys_layout = Mock()
        keys_layout.packed = False
        keys_layout.padded = False

        device = Mock(spec=Device)

        result = _create_composed_mask(bias, seqs_layout, keys_layout, device)

        # Should return None when no masks are needed
        assert result is None
        mock_create_block_mask.assert_not_called()

    def test_unsupported_bias_type(self) -> None:
        """Test that unsupported bias types raise an error."""
        bias = Mock()  # Unknown bias type

        seqs_layout = Mock()
        seqs_layout.packed = False
        seqs_layout.padded = False

        keys_layout = Mock()
        keys_layout.packed = False
        keys_layout.padded = False

        device = Mock(spec=Device)

        with pytest.raises(Exception):  # Should raise NotSupportedError
            _create_composed_mask(bias, seqs_layout, keys_layout, device)
