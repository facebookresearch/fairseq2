# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Adapter for HuggingFace models to fairseq2 CausalLM interface.

This module provides an adapter that wraps HuggingFace transformer models
to make them compatible with fairseq2's CausalLM interface, enabling them
to be used with fairseq2 training recipes like SFT.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from fairseq2.models.clm import CausalLM
from fairseq2.nn import BatchLayout, Embedding, IncrementalStateBag


class _HgEmbeddingWrapper(Embedding):
    """
    Wrapper for HuggingFace embeddings to match fairseq2's Embedding interface.

    HuggingFace uses torch.nn.Embedding which has:
    - num_embeddings
    - padding_idx

    fairseq2 expects:
    - num_embeddings
    - embed_dim
    - pad_idx
    """

    def __init__(self, hf_embedding: torch.nn.Embedding) -> None:
        # Initialize with HF embedding's attributes
        num_embeddings = hf_embedding.num_embeddings
        embed_dim = hf_embedding.embedding_dim
        pad_idx = hf_embedding.padding_idx

        super().__init__(num_embeddings, embed_dim, pad_idx)

        # Store reference to the actual HF embedding
        self._hf_embedding = hf_embedding

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Delegate forward to the HuggingFace embedding."""
        return cast(Tensor, self._hf_embedding(x))

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the HF embedding."""
        # Avoid recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._hf_embedding, name)


class _HgDecoderFrontend:
    """
    Fake decoder frontend that mimics fairseq2's structure.

    This allows HuggingFace models to pass vocabulary checks that expect
    fairseq2's model.decoder_frontend.embed structure.
    """

    def __init__(self, embed_module: Module) -> None:
        self.embed = _HgEmbeddingWrapper(cast(torch.nn.Embedding, embed_module))


class HgCausalLMAdapter(CausalLM):
    """
    Adapter that wraps a HuggingFace causal language model to implement
    fairseq2's CausalLM interface.

    This allows HuggingFace models (like Gemma, Llama, GPT-2, etc.) to be used
    with fairseq2's training recipes that expect the CausalLM interface.

    :param hf_model: The HuggingFace PreTrainedModel to wrap.
    :param max_seq_len: Maximum sequence length supported by the model.
    :param enable_gradient_checkpointing: If ``True``, enables gradient
        checkpointing to save memory.
    """

    _EMBEDDING_PATHS = [
        "embed_tokens",  # Llama, Mistral, etc.
        "model.embed_tokens",  # Some wrapped models
        "wte",  # GPT-2
        "transformer.wte",  # GPT-2 variants
        "embeddings.word_embeddings",  # BERT-style
        "model.embeddings.word_embeddings",
    ]

    def __init__(
        self,
        hf_model: Module,
        max_seq_len: int = 8192,
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(max_seq_len=max_seq_len)

        self.add_module("_wrapped_hf_model", hf_model)

        self._hf_config: Any = getattr(hf_model, "config", None)

        if enable_gradient_checkpointing:
            gc_enable = getattr(hf_model, "gradient_checkpointing_enable", None)
            if gc_enable is None:
                raise RuntimeError(
                    f"Model {type(hf_model).__name__} does not support gradient checkpointing. "
                    f"The model must have a 'gradient_checkpointing_enable()' method."
                )
            try:
                gc_enable()
                # Verify it was actually enabled
                if (
                    hasattr(hf_model, "is_gradient_checkpointing")
                    and not hf_model.is_gradient_checkpointing
                ):
                    raise RuntimeError(
                        f"Gradient checkpointing was called but failed to enable on {type(hf_model).__name__}."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enable gradient checkpointing on {type(hf_model).__name__}: {e}"
                ) from e

        # Create a fake decoder_frontend to mimic fairseq2's structure
        # This allows vocabulary checks to work with HF models
        embed_module = self._find_embedding_layer(hf_model)
        self.decoder_frontend = _HgDecoderFrontend(embed_module)

    def _find_embedding_layer(self, hf_model: Module) -> Module:
        """
        Find the embedding layer in a HuggingFace model.

        Different HF models store embeddings in different places:
        - Most models: model.embed_tokens or model.model.embed_tokens
        - GPT-2: model.wte
        - BERT: model.embeddings.word_embeddings
        """
        # Try common locations for embedding layers
        for attr_path in self._EMBEDDING_PATHS:
            try:
                parts = attr_path.split(".")
                obj = hf_model
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue

        raise LookupError(
            f"Cannot find embedding layer in {type(hf_model).__name__}. "
            f"Tried: {', '.join(self._EMBEDDING_PATHS)}. "
            f"Register the model with a known embedding path or add support for this architecture."
        )

    @property
    def _hf_model(self) -> Module:
        """Access the wrapped HuggingFace model."""
        module = self._modules["_wrapped_hf_model"]
        assert module is not None
        return module

    @override
    def forward(  # type: ignore[override]
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        targets: Tensor | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
        label_smoothing: float = 0.0,
        target_mask: Tensor | None = None,
        reduction: Literal["sum", "mean"] = "sum",
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass that translates fairseq2's interface to HuggingFace's.

        :param seqs: Input token IDs [batch_size, seq_len].
        :param seqs_layout: Layout information (padding, etc.).
        :param targets: Target token IDs for training [batch_size, seq_len].
        :param state_bag: Incremental state (not used for HF models).
        :param label_smoothing: Label smoothing factor (not implemented).
        :param target_mask: Mask for targets [batch_size, seq_len].
        :param reduction: How to reduce the loss (``"sum"`` or ``"mean"``).
        :param return_logits: Whether to return logits along with loss.
        :returns: Loss tensor, or tuple of (loss, logits) if ``return_logits`` is ``True``.
        """
        attention_mask = self._create_attention_mask(seqs, seqs_layout)

        hf_kwargs = {
            "input_ids": seqs,
            "attention_mask": attention_mask,
        }

        # If we have targets, this is a training forward pass
        if targets is not None:
            # HuggingFace models expect labels for training
            # Apply target_mask if provided
            if target_mask is not None:
                # Set ignored positions to -100 (HF's ignore index)
                labels = targets.clone()
                labels = labels.masked_fill(~target_mask, -100)
            else:
                labels = targets

            hf_kwargs["labels"] = labels

            # Forward through HuggingFace model
            outputs = self._hf_model(**hf_kwargs)

            # HuggingFace returns a ModelOutput with .loss and .logits
            loss: Tensor = outputs.loss

            # HF models return mean loss, need to optionally convert to "sum"
            if reduction == "sum":
                if target_mask is not None:
                    num_targets = target_mask.sum()
                else:
                    num_targets = (labels != -100).sum()
                loss = loss * num_targets

            if return_logits:
                return loss, cast(Tensor, outputs.logits)
            else:
                return loss
        else:
            # Inference mode returns logits only
            outputs = self._hf_model(**hf_kwargs)
            return cast(Tensor, outputs.logits)

    def _create_attention_mask(self, seqs: Tensor, seqs_layout: BatchLayout) -> Tensor:
        """
        Create attention mask from fairseq2's BatchLayout.

        :param seqs: Input token IDs [batch_size, seq_len].
        :param seqs_layout: Layout with padding information.
        :returns: Attention mask [batch_size, seq_len] where 1 = attend, 0 = ignore.
        """
        # In fairseq2, position_indices contains -1 for padding positions
        # and valid positions (0, 1, 2, ...) for non-padding positions
        if seqs_layout.padded:
            # Create mask: True for valid positions, False for padding
            padding_mask = seqs_layout.position_indices >= 0
            # HuggingFace expects 1 for real tokens, 0 for padding
            attention_mask = padding_mask.to(dtype=torch.long)
        else:
            # No padding, all positions are valid
            attention_mask = torch.ones_like(seqs, dtype=torch.long)

        return attention_mask

    @property
    def hf_model(self) -> Module:
        """Access the underlying HuggingFace model."""
        return self._hf_model

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying HuggingFace model.
        This allows accessing HF-specific attributes and methods.
        """
        msg = f"'{type(self).__name__}' object has no attribute '{name}'"

        # Avoid infinite recursion for private attributes or during initialization
        if name.startswith("_"):
            raise AttributeError(msg)

        # Check if _wrapped_hf_model has been initialized yet
        # Use object.__getattribute__ to avoid recursion
        try:
            modules = object.__getattribute__(self, "_modules")
            if "_wrapped_hf_model" not in modules:
                raise AttributeError(msg)
        except AttributeError:
            raise AttributeError(msg)

        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the wrapped HF model
            hf_model = modules["_wrapped_hf_model"]
            return getattr(hf_model, name)


def wrap_hg_model_if_causal_lm(hf_model: Module, config: Any) -> Module:
    """
    Wrap a HuggingFace model in HgCausalLMAdapter if it's a causal language model.

    :param hf_model: The HuggingFace model to potentially wrap.
    :param config: The HuggingFaceModelConfig used to create the model.
    :returns: Wrapped model if it's a causal LM, otherwise the original model.
    """
    # Check if this is a causal LM that should be wrapped
    if hasattr(config, "model_type") and config.model_type == "causal_lm":
        # Determine max_seq_len from HF model config
        max_seq_len = 8192  # default
        if hasattr(hf_model, "config"):
            hf_config = hf_model.config
            # Try common attribute names for max position embeddings
            for attr in ["max_position_embeddings", "n_positions", "max_seq_len"]:
                if hasattr(hf_config, attr):
                    max_seq_len = getattr(hf_config, attr)
                    break

        # Check if gradient checkpointing should be enabled
        enable_gradient_checkpointing = getattr(
            config, "enable_gradient_checkpointing", False
        )

        return HgCausalLMAdapter(
            hf_model,
            max_seq_len=max_seq_len,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
        )

    # Not a causal LM
    return hf_model
