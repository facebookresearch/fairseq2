# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.models.hg.adapter import (
    HgCausalLMAdapter,
    _HgDecoderFrontend,
    _HgEmbeddingWrapper,
    wrap_hg_model_if_causal_lm,
)
from fairseq2.nn import BatchLayout


def _make_hf_embedding(
    num_embeddings: int = 50257,
    embedding_dim: int = 768,
    padding_idx: int | None = None,
) -> MagicMock:
    """Create a mock HuggingFace embedding layer."""
    embed = MagicMock(
        spec=["num_embeddings", "embedding_dim", "padding_idx", "forward", "__call__"]
    )
    embed.num_embeddings = num_embeddings
    embed.embedding_dim = embedding_dim
    embed.padding_idx = padding_idx
    return embed


def _make_hf_model(
    embed_attr: str = "embed_tokens",
    num_embeddings: int = 50257,
    embedding_dim: int = 768,
    has_config: bool = True,
    has_gradient_checkpointing: bool = False,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock HuggingFace model with an embedding layer."""
    model = MagicMock(spec=Module)
    model._modules = {}

    embed = _make_hf_embedding(num_embeddings, embedding_dim)

    # Clear all common embedding attributes first
    for attr in ["embed_tokens", "wte", "embeddings"]:
        if hasattr(model, attr):
            delattr(model, attr)

    # Configure mock to only have the specified embedding attribute
    def getattr_side_effect(name: str) -> Any:
        if name == embed_attr:
            return embed
        # For dotted paths, handle the first part
        parts = embed_attr.split(".")
        if name == parts[0] and len(parts) > 1:
            sub = MagicMock()
            obj = sub
            for part in parts[1:-1]:
                next_obj = MagicMock()
                setattr(obj, part, next_obj)
                obj = next_obj
            setattr(obj, parts[-1], embed)
            return sub
        raise AttributeError(f"Mock has no attribute '{name}'")

    model.__getattr__ = getattr_side_effect  # type: ignore[method-assign]
    # Make getattr work through the mock
    model.configure_mock(
        **{embed_attr.split(".")[0]: getattr_side_effect(embed_attr.split(".")[0])}
    )

    if has_config:
        config = MagicMock()
        config.max_position_embeddings = 1024
        model.config = config

    if has_gradient_checkpointing:
        model.gradient_checkpointing_enable = MagicMock()
        model.is_gradient_checkpointing = True

    return model, embed


def _make_simple_hf_model(embed_attr: str = "embed_tokens") -> tuple[Module, Module]:
    """Create a minimal real Module with a real embedding for adapter tests."""

    class FakeHFModel(Module):
        def __init__(self, attr: str) -> None:
            super().__init__()
            self._embed = torch.nn.Embedding(50257, 768)
            # Dynamically set the attribute
            parts = attr.split(".")
            if len(parts) == 1:
                setattr(self, attr, self._embed)
            elif len(parts) == 2:
                sub = Module()
                setattr(sub, parts[1], self._embed)
                setattr(self, parts[0], sub)

        def forward(self, **kwargs: Any) -> Any:
            # Simulate HF model output
            output = MagicMock()
            input_ids: Tensor = kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            if "labels" in kwargs:
                output.loss = torch.tensor(2.5)
                output.logits = torch.randn(batch_size, seq_len, 50257)
            else:
                output.logits = torch.randn(batch_size, seq_len, 50257)
            return output

    model = FakeHFModel(embed_attr)
    return model, model._embed


class TestHgEmbeddingWrapper:
    """Test _HgEmbeddingWrapper."""

    def test_exposes_num_embeddings(self) -> None:
        embed = _make_hf_embedding(num_embeddings=32000)
        wrapper = _HgEmbeddingWrapper(embed)
        assert wrapper.num_embeddings == 32000

    def test_exposes_embed_dim(self) -> None:
        embed = _make_hf_embedding(embedding_dim=4096)
        wrapper = _HgEmbeddingWrapper(embed)
        assert wrapper.embed_dim == 4096

    def test_exposes_pad_idx(self) -> None:
        embed = _make_hf_embedding(padding_idx=0)
        wrapper = _HgEmbeddingWrapper(embed)
        assert wrapper.pad_idx == 0

    def test_pad_idx_none(self) -> None:
        embed = _make_hf_embedding(padding_idx=None)
        wrapper = _HgEmbeddingWrapper(embed)
        assert wrapper.pad_idx is None

    def test_delegates_forward(self) -> None:
        embed = _make_hf_embedding()
        expected = torch.randn(2, 10, 768)
        embed.return_value = expected
        wrapper = _HgEmbeddingWrapper(embed)

        x = torch.randint(0, 50257, (2, 10))
        result = wrapper(x)

        embed.assert_called_once_with(x)
        assert result is expected

    def test_private_attr_raises(self) -> None:
        """Accessing private attrs should raise AttributeError, not recurse."""
        embed = _make_hf_embedding()
        wrapper = _HgEmbeddingWrapper(embed)
        with pytest.raises(AttributeError):
            wrapper._nonexistent


class TestHgDecoderFrontend:
    """Test _HgDecoderFrontend."""

    def test_creates_embedding_wrapper(self) -> None:
        embed = _make_hf_embedding(num_embeddings=32000, embedding_dim=4096)
        frontend = _HgDecoderFrontend(embed)
        assert isinstance(frontend.embed, _HgEmbeddingWrapper)
        assert frontend.embed.num_embeddings == 32000
        assert frontend.embed.embed_dim == 4096


class TestHgCausalLMAdapter:
    """Test HgCausalLMAdapter."""

    def test_init_registers_hf_model_as_submodule(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)
        assert "_wrapped_hf_model" in adapter._modules
        assert adapter._modules["_wrapped_hf_model"] is model

    def test_find_embedding_layer_embed_tokens(self) -> None:
        model, embed = _make_simple_hf_model("embed_tokens")
        adapter = HgCausalLMAdapter(model)
        wrapper = adapter.decoder_frontend.embed
        assert wrapper.num_embeddings == embed.num_embeddings
        assert wrapper.embed_dim == embed.embedding_dim

    def test_find_embedding_layer_wte(self) -> None:
        model, embed = _make_simple_hf_model("wte")
        adapter = HgCausalLMAdapter(model)
        wrapper = adapter.decoder_frontend.embed
        assert wrapper.num_embeddings == embed.num_embeddings
        assert wrapper.embed_dim == embed.embedding_dim

    def test_find_embedding_layer_nested(self) -> None:
        model, embed = _make_simple_hf_model("model.embed_tokens")
        adapter = HgCausalLMAdapter(model)
        wrapper = adapter.decoder_frontend.embed
        assert wrapper.num_embeddings == embed.num_embeddings
        assert wrapper.embed_dim == embed.embedding_dim

    def test_find_embedding_layer_transformer_wte(self) -> None:
        model, embed = _make_simple_hf_model("transformer.wte")
        adapter = HgCausalLMAdapter(model)
        wrapper = adapter.decoder_frontend.embed
        assert wrapper.num_embeddings == embed.num_embeddings
        assert wrapper.embed_dim == embed.embedding_dim

    def test_find_embedding_layer_fallback(self) -> None:
        """Model with no known embedding attr raises LookupError."""

        class NoEmbedModel(Module):
            def forward(self, **kwargs: Any) -> None:
                pass

        model = NoEmbedModel()
        with pytest.raises(LookupError, match="Cannot find embedding layer"):
            HgCausalLMAdapter(model)

    def test_gradient_checkpointing_enable_fails_verification(self) -> None:
        """Test RuntimeError when gradient checkpointing call succeeds but verification fails."""

        class BadGCModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)

            def gradient_checkpointing_enable(self) -> None:
                pass  # Does nothing

            @property
            def is_gradient_checkpointing(self) -> bool:
                return False  # Always reports disabled

            def forward(self, **kwargs: Any) -> None:
                pass

        model = BadGCModel()
        with pytest.raises(RuntimeError, match="failed to enable"):
            HgCausalLMAdapter(model, enable_gradient_checkpointing=True)

    def test_gradient_checkpointing_enable_raises(self) -> None:
        """Test RuntimeError when gradient_checkpointing_enable() itself throws."""

        class ExplodingGCModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)

            def gradient_checkpointing_enable(self) -> None:
                raise ValueError("CUDA required")

            def forward(self, **kwargs: Any) -> None:
                pass

        model = ExplodingGCModel()
        with pytest.raises(
            RuntimeError, match="Failed to enable gradient checkpointing"
        ):
            HgCausalLMAdapter(model, enable_gradient_checkpointing=True)

    def test_getattr_private_attr_raises(self) -> None:
        """Test that accessing private attrs raises AttributeError, not infinite recursion."""
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)
        with pytest.raises(AttributeError):
            adapter._nonexistent_private

    def test_getattr_missing_public_attr_raises(self) -> None:
        """Test that accessing non-existent public attrs raises AttributeError."""

        class MinimalModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)

            def forward(self, **kwargs: Any) -> None:
                pass

        model = MinimalModel()
        adapter = HgCausalLMAdapter(model)
        with pytest.raises(AttributeError):
            adapter.totally_nonexistent_attribute

    def test_forward_training_with_targets(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)

        batch_size, seq_len = 2, 10
        seqs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        layout = BatchLayout((batch_size, seq_len), seq_lens=None)

        loss = adapter(seqs, layout, targets=targets)

        assert isinstance(loss, Tensor)
        assert loss.dim() == 0  # scalar

    def test_forward_training_with_target_mask(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)

        batch_size, seq_len = 2, 10
        seqs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        # Mask out last 3 positions
        target_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        target_mask[:, -3:] = False
        layout = BatchLayout((batch_size, seq_len), seq_lens=None)

        loss = adapter(seqs, layout, targets=targets, target_mask=target_mask)

        assert isinstance(loss, Tensor)
        assert loss.dim() == 0

    def test_forward_reduction_sum_vs_mean(self) -> None:
        """Verify sum reduction multiplies mean loss by number of targets."""

        class FixedLossModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)

            def forward(self, **kwargs: Any) -> Any:
                output = MagicMock()
                input_ids = kwargs["input_ids"]
                b, s = input_ids.shape
                output.loss = torch.tensor(1.0)  # fixed mean loss
                output.logits = torch.randn(b, s, 100)
                return output

        model = FixedLossModel()
        adapter = HgCausalLMAdapter(model)

        batch_size, seq_len = 2, 5
        seqs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        layout = BatchLayout((batch_size, seq_len), seq_lens=None)

        loss_sum = adapter(seqs, layout, targets=targets, reduction="sum")
        loss_mean = adapter(seqs, layout, targets=targets, reduction="mean")

        # Mean loss should be 1.0 (as returned by model)
        torch.testing.assert_close(loss_mean, torch.tensor(1.0))
        # Sum loss = mean_loss * num_targets = 1.0 * 10 = 10.0
        expected_sum = torch.tensor(float(batch_size * seq_len))
        torch.testing.assert_close(loss_sum, expected_sum)

    def test_forward_inference_without_targets(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)

        batch_size, seq_len = 2, 10
        seqs = torch.randint(0, 100, (batch_size, seq_len))
        layout = BatchLayout((batch_size, seq_len), seq_lens=None)

        logits = adapter(seqs, layout)

        assert isinstance(logits, Tensor)
        assert logits.shape == (batch_size, seq_len, 50257)

    def test_forward_return_logits_flag(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)

        batch_size, seq_len = 2, 10
        seqs = torch.randint(0, 100, (batch_size, seq_len))
        targets = torch.randint(0, 100, (batch_size, seq_len))
        layout = BatchLayout((batch_size, seq_len), seq_lens=None)

        result = adapter(seqs, layout, targets=targets, return_logits=True)

        assert isinstance(result, tuple)
        loss, logits = result
        assert loss.dim() == 0
        assert logits.shape == (batch_size, seq_len, 50257)

    def test_create_attention_mask_padded(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)

        batch_size, max_seq_len = 2, 10
        seqs = torch.randint(0, 100, (batch_size, max_seq_len))
        # First sequence has length 7, second has length 10
        layout = BatchLayout((batch_size, max_seq_len), seq_lens=[7, 10])

        mask = adapter._create_attention_mask(seqs, layout)

        assert mask.shape == (batch_size, max_seq_len)
        assert mask.dtype == torch.long
        # First sequence: 7 ones, 3 zeros
        assert mask[0, :7].sum() == 7
        assert mask[0, 7:].sum() == 0
        # Second sequence: all ones
        assert mask[1].sum() == 10

    def test_create_attention_mask_unpadded(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)

        batch_size, seq_len = 2, 10
        seqs = torch.randint(0, 100, (batch_size, seq_len))
        layout = BatchLayout((batch_size, seq_len), seq_lens=None)

        mask = adapter._create_attention_mask(seqs, layout)

        assert mask.shape == (batch_size, seq_len)
        assert (mask == 1).all()

    def test_gradient_checkpointing_enable(self) -> None:
        class GCModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)
                self._gc_enabled = False

            def gradient_checkpointing_enable(self) -> None:
                self._gc_enabled = True

            @property
            def is_gradient_checkpointing(self) -> bool:
                return self._gc_enabled

            def forward(self, **kwargs: Any) -> None:
                pass

        model = GCModel()
        HgCausalLMAdapter(model, enable_gradient_checkpointing=True)
        assert model._gc_enabled

    def test_gradient_checkpointing_unsupported(self) -> None:
        class NoGCModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)

            def forward(self, **kwargs: Any) -> None:
                pass

        model = NoGCModel()
        with pytest.raises(
            RuntimeError, match="does not support gradient checkpointing"
        ):
            HgCausalLMAdapter(model, enable_gradient_checkpointing=True)

    def test_getattr_delegates_to_hf_model(self) -> None:
        class CustomModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(100, 32)
                self.custom_attr = "hello"

            def forward(self, **kwargs: Any) -> None:
                pass

        model = CustomModel()
        adapter = HgCausalLMAdapter(model)
        assert adapter.custom_attr == "hello"

    def test_hf_model_property(self) -> None:
        model, _ = _make_simple_hf_model()
        adapter = HgCausalLMAdapter(model)
        assert adapter.hf_model is model


class TestWrapHgModelIfCausalLm:
    """Test wrap_hg_model_if_causal_lm."""

    def test_wraps_causal_lm_config(self) -> None:
        model, _ = _make_simple_hf_model()
        config = MagicMock()
        config.model_type = "causal_lm"
        config.enable_gradient_checkpointing = False

        result = wrap_hg_model_if_causal_lm(model, config)

        assert isinstance(result, HgCausalLMAdapter)

    def test_passes_through_non_causal(self) -> None:
        model, _ = _make_simple_hf_model()
        config = MagicMock()
        config.model_type = "seq2seq_lm"

        result = wrap_hg_model_if_causal_lm(model, config)

        assert result is model
        assert not isinstance(result, HgCausalLMAdapter)
