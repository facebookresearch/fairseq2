# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast, final

from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
)
from typing_extensions import override

from fairseq2.error import InternalError, NotSupportedError
from fairseq2.models.transformer.attention_bias import (
    AttentionBias,
    AttentionBiasCache,
    CausalAttentionBias,
    IdentityBias,
)
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import BatchLayout

MaskMod: TypeAlias = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


@final
class FlexSDPA(SDPA):
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
        if needs_weights:
            raise NotSupportedError(f"`{FlexSDPA}` does not support `needs_weights`.")

        block_mask = self._maybe_get_block_mask(seqs_layout, keys_layout, bias_cache)

        if not self.training:
            dropout_p = 0.0
        else:
            dropout_p = self.dropout_p

        if dropout_p != 0.0:
            raise NotSupportedError(f"`{FlexSDPA}` does not support dropout.")

        q, k, v = seqs, keys, values

        # (N, S, H, K) -> (N, H, S, K)
        q = q.transpose(-2, -3)

        # (N, S_kv, H, K) -> (N, H, S_kv, K)
        k = k.transpose(-2, -3)

        # (N, S_kv, H, V) -> (N, H, S_kv, V)
        v = v.transpose(-2, -3)

        attns = flex_attention(seqs, keys, values, block_mask=block_mask)

        attns = cast(Tensor, attns)

        # (N, H, S, V) -> (N, S, H, V)
        attns = attns.transpose(-2, -3)

        return attns, None

    def _maybe_get_block_mask(
        self,
        seqs_layout: BatchLayout,
        keys_layout: BatchLayout,
        bias_cache: AttentionBiasCache,
    ) -> BlockMask | None:
        return None

    #        block_mask = attn_mask.cache.get("flex")
    #        if block_mask is not None:
    #            if not isinstance(block_mask, BlockMask):
    #                raise InternalError(
    #                    f"`attn_mask.data['flex'] is of type `{type(block_mask)}`."
    #                )
    #
    #            return block_mask
    #
    #        if key_padding_mask is not None:
    #            raise NotSupportedError("`FlexSDPA` does not support `key_padding_mask`.")
    #
    #        attn_kind = attn_mask.attn_kind
    #
    #        if isinstance(attn_kind, FullAttentionKind):
    #            return None
    #
    #        if not isinstance(attn_kind, CausalAttentionKind):
    #            raise NotSupportedError(f"`FlexSDPA` does not support `{attn_kind}`.")
    #
    #        def causal_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    #            return q_idx >= kv_idx
    #
    #        mask_mod: MaskMod
    #
    #        attn_window_len = attn_kind.attn_window_len
    #        if attn_window_len is None:
    #            mask_mod = causal_mod
    #        else:
    #
    #            def sliding_window_mod(
    #                b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor
    #            ) -> Tensor:
    #                return q_idx - kv_idx <= attn_window_len
    #
    #            mask_mod = and_masks(causal_mod, sliding_window_mod)
    #
    #        seq_len = seqs.size(2)
    #        key_len = keys.size(2)
    #
    #        block_mask = create_block_mask(
    #            mask_mod,
    #            B=None,
    #            H=None,
    #            Q_LEN=seq_len,
    #            KV_LEN=key_len,
    #            device=seqs.device,  # type: ignore[arg-type]
    #        )
    #
    #        attn_mask.cache["flex"] = block_mask
    #
    #        return block_mask

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"bias={self.bias}, dropout_p={self.dropout_p:G}"
