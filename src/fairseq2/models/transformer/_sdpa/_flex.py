# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Callable, cast, final

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention, or_masks
from typing_extensions import override

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


def _causal_mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    """Standard causal attention mask."""
    return q_idx >= kv_idx


def _sliding_window_causal_mask_fn(window_size: int) -> Callable:
    """Creates a sliding window causal mask function."""
    def mask_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        return (q_idx >= kv_idx) and (q_idx - kv_idx <= window_size)
    return mask_fn


def _dropout_mask_fn(dropout_p: float, training: bool = True) -> Callable | None:
    """Creates a dropout mask function."""
    if not training or dropout_p == 0.0:
        return None
   
    def dropout_fn(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        # Generate deterministic random number based on position
        generator = torch.Generator()
        generator.manual_seed(hash((b, h, q_idx, kv_idx)) % (2**32))
        rand_val = torch.rand(1, generator=generator).item()
       
        # Return True to keep, False to mask (opposite of dropout probability)
        return rand_val >= dropout_p
   
    return dropout_fn


def _create_composed_mask(bias: AttentionBias, dropout_p: float = 0.0, training: bool = True) -> Callable | None:
    """Creates a composed mask using or_mask for combining multiple mask functions."""
   
    masks = []
   
    # Add attention bias mask
    if isinstance(bias, CausalAttentionBias):
        attn_window_len = bias.attn_window_len
        if attn_window_len is not None:
            masks.append(_sliding_window_causal_mask_fn(attn_window_len))
        else:
            masks.append(_causal_mask_fn)
    elif not isinstance(bias, IdentityBias):
        raise NotSupportedError(f"Unsupported bias type: {bias}")
   
    # Add dropout mask if needed
    dropout_mask = _dropout_mask_fn(dropout_p, training)
    if dropout_mask is not None:
        masks.append(dropout_mask)
   
    # Compose masks using or_mask
    if len(masks) == 0:
        return None
    elif len(masks) == 1:
        return masks[0]
    else:
        # Use or_mask to combine multiple masks
        return or_masks(*masks)


@final
class FlexSDPA(SDPA):
    """Computes scaled dot-product attention using PyTorch's Flex Attention."""

    bias: AttentionBias
    dropout_p: float

    def __init__(self, bias: AttentionBias, *, dropout_p: float = 0.0) -> None:
        super().__init__(bias)

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
        if seqs_layout.padded or keys_layout.padded:
            raise NotSupportedError(f"`{FlexSDPA}` does not support padded batches.")

        if seqs_layout.packed ^ keys_layout.packed:
            raise ValueError("`seqs_layout` and `keys_layout` must be both packed.")

        # Handle dropout
        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        # Create the composed mask using or_mask for clean composition
        mask_fn = _create_composed_mask(self.bias, dropout_p, self.training)

        if seqs_layout.packed:
            # For packed sequences, we need to handle variable length sequences
            # This is more complex with Flex Attention and may require custom handling
            # For now, we'll reshape and use the standard flex_attention
            batch_size = len(seqs_layout.seq_begin_indices_pt) - 1
            max_seq_len = seqs_layout.max_seq_len
            num_heads = seqs.size(1)
            head_dim = seqs.size(-1)
           
            # Reshape from packed format to batch format
            # This is a simplified approach - in practice you'd need more sophisticated
            # handling of variable length sequences
            seqs_batch = seqs.new_zeros(batch_size, num_heads, max_seq_len, head_dim)
            keys_batch = keys.new_zeros(batch_size, num_heads, max_seq_len, head_dim)
            values_batch = values.new_zeros(batch_size, num_heads, max_seq_len, head_dim)
           
            for i in range(batch_size):
                start_idx = seqs_layout.seq_begin_indices_pt[i]
                end_idx = seqs_layout.seq_begin_indices_pt[i + 1]
                seq_len = end_idx - start_idx
               
                seqs_batch[i, :, :seq_len] = seqs[start_idx:end_idx].transpose(0, 1)
                keys_batch[i, :, :seq_len] = keys[start_idx:end_idx].transpose(0, 1)
                values_batch[i, :, :seq_len] = values[start_idx:end_idx].transpose(0, 1)
           
            # Apply flex attention with composed mask
            attns_batch = flex_attention(
                seqs_batch, keys_batch, values_batch,
                score_mod=mask_fn,
                enable_gqa=False
            )
           
            # Convert back to packed format
            total_len = seqs.size(0)
            attns = seqs.new_zeros(total_len, num_heads, head_dim)
           
            for i in range(batch_size):
                start_idx = seqs_layout.seq_begin_indices_pt[i]
                end_idx = seqs_layout.seq_begin_indices_pt[i + 1]
                seq_len = end_idx - start_idx
               
                attns[start_idx:end_idx] = attns_batch[i, :, :seq_len].transpose(0, 1)
        else:
            # Standard batch format
            attns = flex_attention(
                seqs,
                keys,
                values,
                score_mod=mask_fn,
                enable_gqa=False
            )

        attns = cast(Tensor, attns)

        return attns, None

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, dropout_p={self.dropout_p:G}"