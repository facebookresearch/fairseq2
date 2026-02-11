# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from fairseq2.models.gemma3n.kv_sharing import KVSharedLayerRegistry
from tests.common import device


class TestKVSharedLayerRegistry:
    """Test KV projection sharing mechanism."""

    def test_store_and_retrieve_kv(self) -> None:
        """Verify source layer stores and consumer retrieves K/V correctly."""
        registry = KVSharedLayerRegistry()

        # Create mock K/V tensors (B, H, S, D)
        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 32
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Source layer stores K/V
        source_layer_idx = 18
        registry.store_kv_for_sharing(source_layer_idx, key, value)

        # Consumer layer retrieves K/V
        consumer_layer_idx = 20
        retrieved_key, retrieved_value = registry.retrieve_shared_kv(
            consumer_layer_idx, source_layer_idx
        )

        # Verify exact same tensors returned
        assert torch.equal(retrieved_key, key)
        assert torch.equal(retrieved_value, value)
        assert retrieved_key.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_multiple_source_layers(self) -> None:
        """Verify multiple source layers can store independently."""
        registry = KVSharedLayerRegistry()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16

        # Store from two different source layers
        k1 = torch.ones(batch_size, num_heads, seq_len, head_dim, device=device)
        v1 = torch.ones(batch_size, num_heads, seq_len, head_dim, device=device) * 2

        k2 = torch.ones(batch_size, num_heads, seq_len, head_dim, device=device) * 3
        v2 = torch.ones(batch_size, num_heads, seq_len, head_dim, device=device) * 4

        registry.store_kv_for_sharing(18, k1, v1)  # Local source
        registry.store_kv_for_sharing(19, k2, v2)  # Global source

        # Retrieve from each
        ret_k1, ret_v1 = registry.retrieve_shared_kv(20, 18)
        ret_k2, ret_v2 = registry.retrieve_shared_kv(24, 19)

        # Verify correct retrieval
        assert torch.equal(ret_k1, k1)
        assert torch.equal(ret_v1, v1)
        assert torch.equal(ret_k2, k2)
        assert torch.equal(ret_v2, v2)

    def test_duplicate_store_raises_error(self) -> None:
        """Verify storing K/V twice from same source layer raises RuntimeError."""
        registry = KVSharedLayerRegistry()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        source_layer_idx = 18
        registry.store_kv_for_sharing(source_layer_idx, key, value)

        # Attempt to store again should fail
        with pytest.raises(RuntimeError, match="already stored K/V"):
            registry.store_kv_for_sharing(source_layer_idx, key, value)

    def test_retrieve_before_store_raises_error(self) -> None:
        """Verify retrieving before source stores raises RuntimeError."""
        registry = KVSharedLayerRegistry()

        # Consumer tries to retrieve before source stores
        with pytest.raises(RuntimeError, match="has not stored K/V yet"):
            registry.retrieve_shared_kv(consumer_layer_idx=20, source_layer_idx=18)

    def test_clear_resets_registry(self) -> None:
        """Verify clear() resets registry for next forward pass."""
        registry = KVSharedLayerRegistry()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Store K/V
        registry.store_kv_for_sharing(18, key, value)

        # Clear registry
        registry.clear()

        # After clear, retrieve should fail
        with pytest.raises(RuntimeError, match="has not stored K/V yet"):
            registry.retrieve_shared_kv(20, 18)

        # Should be able to store again after clear
        registry.store_kv_for_sharing(18, key, value)
        ret_k, ret_v = registry.retrieve_shared_kv(20, 18)
        assert torch.equal(ret_k, key)

    def test_repr_shows_source_layers(self) -> None:
        """Verify __repr__ shows stored source layers."""
        registry = KVSharedLayerRegistry()

        batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 16
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Initially empty
        repr_str = repr(registry)
        assert "source_layers=[]" in repr_str

        # After storing
        registry.store_kv_for_sharing(18, key, value)
        registry.store_kv_for_sharing(19, key, value)

        repr_str = repr(registry)
        assert "[18, 19]" in repr_str

    def test_shape_preservation(self) -> None:
        """Verify K/V shapes preserved through store/retrieve cycle."""
        registry = KVSharedLayerRegistry()

        # Test various shapes
        test_shapes = [
            (1, 2, 4, 16),  # Minimal
            (2, 8, 128, 64),  # Typical
            (4, 16, 512, 32),  # Large seq_len
        ]

        for batch_size, num_heads, seq_len, head_dim in test_shapes:
            registry.clear()
            key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            value = torch.randn(
                batch_size, num_heads, seq_len, head_dim, device=device
            )

            registry.store_kv_for_sharing(0, key, value)
            ret_k, ret_v = registry.retrieve_shared_kv(1, 0)

            assert ret_k.shape == key.shape
            assert ret_v.shape == value.shape
            assert ret_k.dtype == key.dtype
            assert ret_v.dtype == value.dtype
