# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, TypeAlias, final

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
)
from typing_extensions import override

from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.nn import BatchLayout

# isort: split

from fairseq2.models.transformer._attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer._sdpa._base import SDPA

MaskFunction: TypeAlias = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def _causal_mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    """Standard causal attention mask."""
    return q_idx >= kv_idx


def _sliding_window_causal_mask_fn(window_size: int) -> MaskFunction:
    """Creates a sliding window causal mask function."""

    def mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= window_size)

    return mask_fn


def _offsets_to_doc_ids_tensor(offsets: Tensor) -> Tensor:
    """Convert offsets to document IDs for packed sequences."""
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def _create_packed_mask_fn(
    seq_begin_indices: Tensor,
    keys_begin_indices: Tensor,
    base_mask_fn: MaskFunction | None = None,
) -> MaskFunction:
    """Creates a mask function for packed sequences using document-based masking."""
    # Create document IDs for queries and keys
    query_doc_ids = _offsets_to_doc_ids_tensor(seq_begin_indices)
    key_doc_ids = _offsets_to_doc_ids_tensor(keys_begin_indices)

    def packed_mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        # Check if query and key belong to the same document
        same_doc = query_doc_ids[q_idx] == key_doc_ids[kv_idx]

        # Convert global indices to logical positions within documents
        q_doc_id = query_doc_ids[q_idx]
        kv_doc_id = key_doc_ids[kv_idx]
        q_logical = q_idx - seq_begin_indices[q_doc_id]
        kv_logical = kv_idx - keys_begin_indices[kv_doc_id]

        # Apply base mask (e.g., causal) to logical positions
        if base_mask_fn is not None:
            inner_mask = base_mask_fn(b, h, q_logical, kv_logical)
            return same_doc & inner_mask
        else:
            return same_doc

    return packed_mask_fn


def _create_padding_mask_fn(seq_lens: Tensor, value_seq_lens: Tensor) -> MaskFunction:
    """Creates a padding mask function that masks out padding tokens."""

    def padding_mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_valid = q_idx < seq_lens[b]
        kv_valid = kv_idx < value_seq_lens[b]
        return q_valid & kv_valid

    return padding_mask_fn


def _dropout_mask_fn(dropout_p: float, training: bool = True) -> MaskFunction | None:
    """Creates a dropout mask function."""
    if not training or dropout_p == 0.0:
        return None

    def dropout_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        generator = torch.Generator()  # TODO: How to set seed?
        rand_val = torch.rand(1, generator=generator)

        # Return True to keep, False to mask (opposite of dropout probability)
        return rand_val >= dropout_p

    return dropout_fn


def _create_composed_mask(
    bias: AttentionBias,
    seqs_layout: BatchLayout,
    keys_layout: BatchLayout,
    device: Device,
    dropout_p: float = 0.0,
    training: bool = True,
    packed: bool = False,
) -> BlockMask | None:
    """Creates a composed mask using and_mask for combining multiple mask functions."""
    masks = []

    if packed:
        # For packed sequences, create the base mask function first
        base_mask_fn = None

        # Add attention bias mask as base mask
        if isinstance(bias, CausalAttentionBias):
            attn_window_len = bias.attn_window_len
            if attn_window_len is not None:
                base_mask_fn = _sliding_window_causal_mask_fn(attn_window_len)
            else:
                base_mask_fn = _causal_mask_fn
        elif not isinstance(bias, IdentityBias):
            raise NotSupportedError(f"Unsupported bias type: {bias}")

        # Create the packed sequence mask that incorporates the base mask
        packed_mask = _create_packed_mask_fn(
            seqs_layout.seq_begin_indices_pt,
            keys_layout.seq_begin_indices_pt,
            base_mask_fn,
        )
        masks.append(packed_mask)
    else:
        # Standard batch format - handle bias and padding separately
        if isinstance(bias, CausalAttentionBias):
            attn_window_len = bias.attn_window_len
            if attn_window_len is not None:
                masks.append(_sliding_window_causal_mask_fn(attn_window_len))
            else:
                masks.append(_causal_mask_fn)
        elif not isinstance(bias, IdentityBias):
            raise NotSupportedError(f"Unsupported bias type: {bias}")

        # Add padding mask
        masks.append(
            _create_padding_mask_fn(seqs_layout.seq_lens_pt, keys_layout.seq_lens_pt)
        )

    # Add dropout mask if needed
    dropout_mask = _dropout_mask_fn(dropout_p, training)
    if dropout_mask is not None:
        masks.append(dropout_mask)

    # Compose masks using and_mask
    mask_fn = None
    if len(masks) == 0:
        return None
    elif len(masks) == 1:
        mask_fn = masks[0]
    else:
        mask_fn = and_masks(*masks)

    # For packed sequences, use the total sequence length
    if packed:
        total_seq_len = int(seqs_layout.seq_begin_indices_pt[-1].item())
        total_keys_len = int(keys_layout.seq_begin_indices_pt[-1].item())
        batch_size = 1  # Packed format treats everything as one big batch
    else:
        total_seq_len = seqs_layout.max_seq_len
        total_keys_len = keys_layout.max_seq_len
        batch_size = seqs_layout.width

    block_mask = create_block_mask(
        mask_fn,
        B=batch_size,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_keys_len,
        device=str(device),
    )
    return block_mask


@final
class FlexSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch's Flex Attention."""

    bias: AttentionBias
    dropout_p: float

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__()

        self.bias = bias
        self.dropout_p = dropout_p

    @override
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        needs_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if seqs_layout.packed ^ keys_layout.packed:
            raise ValueError("`seqs_layout` and `keys_layout` must be both packed.")

        # Handle dropout
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        # Create the composed block mask using and_mask
        block_mask = _create_composed_mask(
            self.bias,
            seqs_layout,
            keys_layout,
            seqs.device,
            dropout_p,
            self.training,
            packed=seqs_layout.packed,
        )

        seqs = seqs.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attns = flex_attention(
            seqs,
            keys,
            values,
            block_mask=block_mask,
            enable_gqa=False,
        )

        if isinstance(attns, tuple):
            attns, _ = attns

        attns = attns.transpose(1, 2)

        return attns, None

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, dropout_p={self.dropout_p:G}"
