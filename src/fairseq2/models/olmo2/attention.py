# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMO2-specific attention module with Q/K normalization applied before reshaping.

Note: OLMO2MultiheadAttention inherits from StandardMultiheadAttention (marked @final)
because the only difference is the order of normalization in _project_q() and _project_kv().
Reimplementing the entire class would duplicate ~150 lines of boilerplate code for __init__,
projection setup, and forward logic. The type checker warning is suppressed as this is a
legitimate architectural need specific to OLMO2's design.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.gang import Gangs
from fairseq2.models.transformer import StandardMultiheadAttention
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import (
    BatchLayout,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
)


class OLMO2MultiheadAttention(StandardMultiheadAttention):  # type: ignore[misc]
    """OLMO2 Multi-head Attention with Q/K normalization applied BEFORE reshaping.

    The key difference from StandardMultiheadAttention is the order of operations:
    - Standard: Project → Reshape → Normalize → RoPE
    - OLMO2:    Project → Normalize → Reshape → RoPE

    This is why OLMO2's Q/K norm weights have shape [2048] (full projection) instead
    of [128] (head_dim).
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        sdpa: SDPA,
        *,
        head_dim: int | None = None,
        num_key_value_heads: int | None = None,
        kv_dim: int | None = None,
        q_proj: Projection | None = None,
        k_proj: Projection | None = None,
        v_proj: Projection | None = None,
        qkv_proj_init_fn: Callable[[Linear], None] | None = None,
        q_norm: LayerNorm | None = None,
        k_norm: LayerNorm | None = None,
        pos_encoder: PositionEncoder | None = None,
        output_proj: Projection | None = None,
        output_proj_init_fn: Callable[[Linear], None] | None = None,
        bias: bool = True,
        output_proj_bias: bool | None = None,
        state_factory: Any = None,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """Initialize OLMO2 Multi-head Attention.

        All parameters are passed to StandardMultiheadAttention, but the normalization
        order is different.
        """
        super().__init__(
            model_dim=model_dim,
            num_heads=num_heads,
            sdpa=sdpa,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            kv_dim=kv_dim,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            qkv_proj_init_fn=qkv_proj_init_fn,
            q_norm=q_norm,
            k_norm=k_norm,
            pos_encoder=pos_encoder,
            output_proj=output_proj,
            output_proj_init_fn=output_proj_init_fn,
            bias=bias,
            output_proj_bias=output_proj_bias,
            state_factory=state_factory,
            gangs=gangs,
            device=device,
            dtype=dtype,
        )

    @override
    def _project_q(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, K_proj)
        q = self.q_proj(seqs)

        # OLMO2-specific: Apply normalization BEFORE reshaping
        if self.q_norm is not None:
            q = self.q_norm(q)

        # Reshape (N, S, K_proj) -> (N, S, H, K_h)
        q = q.unflatten(-1, (-1, self.head_dim))

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, seqs_layout, state_bag=state_bag)

        return q

    @override
    def _project_kv(
        self,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, Tensor]:
        # (N, S, K) -> (N, S, K_proj)
        k = self.k_proj(keys)
        # (N, S, V) -> (N, S, V_proj)
        v = self.v_proj(values)

        # OLMO2-specific: Apply normalization BEFORE reshaping
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Reshape (N, S, K_proj) -> (N, S, H, K_h)
        k = k.unflatten(-1, (-1, self.head_dim))
        # Reshape (N, S, V_proj) -> (N, S, H, V_h)
        v = v.unflatten(-1, (-1, self.head_dim))

        if self.pos_encoder is not None:
            k = self.pos_encoder(k, keys_layout, state_bag=state_bag)

        return k, v
