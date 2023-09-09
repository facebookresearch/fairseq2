# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final

from torch import Tensor

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer.attention import SDPA, create_default_sdpa
from fairseq2.nn.transformer.multihead_attention import (
    AttentionOutputProjection,
    MultiheadAttention,
    MultiheadAttentionState,
    QKVProjection,
)
from fairseq2.typing import DataType, Device, finaloverride


@final
class GroupedQueryAttention(MultiheadAttention):
    """Represents a Grouped Query Attention attention as described in
    :cite:t:`https://doi.org/10.48550/arXiv.2305.13245`."""

    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    pos_encoder: Optional[PositionEncoder]
    sdpa: SDPA
    output_proj: Projection

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        q_proj: Optional[Projection] = None,
        k_proj: Optional[Projection] = None,
        v_proj: Optional[Projection] = None,
        pos_encoder: Optional[PositionEncoder] = None,
        sdpa: Optional[SDPA] = None,
        output_proj: Optional[Projection] = None,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        :param num_key_value_heads:
            The number of key-value heads for Grouped Query Attention.
            If set to ``num_attn_heads``, it is equivalent to normal
            Multi Head Attention (MHA). If set to ``1``, it is equivalent to
            Multi Query Attention (MQA).
        :param q_proj:
            The projection to apply to queries before computing attention. If
            ``None``, a default projection will be used.
        :param k_proj:
            The projection to apply to keys before computing attention. If
            ``None``, a default projection will be used.
        :param v_proj:
            The projection to apply to values before computing attention. If
            ``None``, a default projection will be used.
        :param pos_encoder:
            The position encoder to apply to queries and keys after projection.
        :param sdpa:
            The scaled dot-product attention module to compute head attentions.
            If ``None``, a default implementation will be used.
        :param scale_heads:
            If ``True``, applies head scaling as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`
        :param output_proj:
            The projection to produce final attentions. If ``None``, a
            default projection will be used.
        :param bias:
            If ``True``, query, key, value, and output projections learn an
            additive bias. Ignored for explicitly specified projections.
        """
        super().__init__(model_dim, num_heads)

        if model_dim % num_heads != 0:
            raise ValueError(
                f"`model_dim` ({model_dim}) must be divisible by `num_heads` ({num_heads})."
            )

        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"`num_heads` ({num_heads}) must be divisible by `num_key_value_heads` ({num_key_value_heads})."
            )

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = model_dim // num_heads
        self.num_key_value_groups = num_heads // num_key_value_heads

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = QKVProjection(
                self.head_dim * num_heads, bias, device=device, dtype=dtype
            )
            k_proj = QKVProjection(
                self.head_dim * num_key_value_heads, bias, device=device, dtype=dtype
            )
            v_proj = QKVProjection(
                self.head_dim * num_key_value_heads, bias, device=device, dtype=dtype
            )
        else:
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError(
                    "`q_proj`, `k_proj`, and `v_proj` must be all specified."
                )

            if q_proj.input_dim != model_dim:
                raise ValueError(
                    f"`input_dim` of `q_proj` must be equal to `model_dim` ({model_dim}), but is {q_proj.input_dim} instead."
                )

            if k_proj.output_dim % num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `k_proj` must be divisible by `num_key_value_heads` ({num_key_value_heads}), but is {k_proj.output_dim} instead."
                )

            if v_proj.output_dim % num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `v_proj` must be divisible by `num_key_value_heads` ({num_key_value_heads}), but is {v_proj.output_dim} instead."
                )

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        if pos_encoder is not None:
            if self.head_dim != pos_encoder.encoding_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and the size of the header key dimension must be equal, but are {pos_encoder.encoding_dim} and {self.head_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if sdpa is not None:
            self.sdpa = sdpa
        else:
            self.sdpa = create_default_sdpa()

        if output_proj is None:
            self.output_proj = AttentionOutputProjection(
                model_dim, model_dim, bias, device=device, dtype=dtype
            )
        else:
            if output_proj.input_dim != model_dim:
                raise ValueError(
                    f"`input_dim` of `output_proj` must be equal to `model_dim` ({model_dim}), but is {output_proj.input_dim} instead."
                )

            if output_proj.output_dim != model_dim:
                raise ValueError(
                    f"`output_dim` of `output_proj` must be equal to `model_dim` ({model_dim}), but is {output_proj.output_dim} instead."
                )

            self.output_proj = output_proj

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        padding_mask: Optional[Tensor],
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)

        if not self.training and state_bag is not None:
            state = state_bag.get_state(self, MultiheadAttentionState)
            if state is not None:
                k, v, key_padding_mask = state.append(k, v, key_padding_mask)
            else:
                state_bag.set_state(
                    self, MultiheadAttentionState(k, v, key_padding_mask)
                )

        # (N, S, model_dim) -> (N, S, H, head_dim)
        q = q.unflatten(-1, (self.num_heads, -1))
        # (N, S_kv, K_proj) -> (N, S_kv, H_kv, head_dim)
        k = k.unflatten(-1, (self.num_key_value_heads, -1))
        # (N, S_kv, V_proj) -> (N, S_kv, H_kv, head_dim)
        v = v.unflatten(-1, (self.num_key_value_heads, -1))

        # (N, S, H, head_dim) -> (N, H, S, head_dim)
        q = q.transpose(1, 2)
        # (N, S_kv, H_kv, head_dim) -> (N, H_kv, S_kv, head_dim)
        k = k.transpose(1, 2)
        # (N, S_kv, H_kv, head_dim) -> (N, H_kv, S_kv, head_dim)
        v = v.transpose(1, 2)

        # (N, H, S, head_dim) -> (N x H, S, head_dim)
        q = q.flatten(0, 1)
        # (N, S_kv, H_kv, head_dim) -> (N x H_kv, S_kv, head_dim)
        k = k.flatten(0, 1)
        # (N, S_kv, H_kv, head_dim) -> (N x H_kv, S_kv, head_dim)
        v = v.flatten(0, 1)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, padding_mask, state_bag)
            k = self.pos_encoder(k, key_padding_mask)

        if key_padding_mask is not None:
            # (N, S_kv) -> (N, 1, 1, S_kv)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            # (N, 1, 1, S_kv) -> (N, H, 1, S_kv)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)

            if attn_mask is None:
                # (N, H, 1, S_kv)
                attn_mask = key_padding_mask
            else:
                # (N, H, 1, S_kv) + ([H,], S, S_kv) = (N, H, S, S_kv)
                attn_mask = key_padding_mask + attn_mask

            # (N, H, S, S_kv) -> (N x H, 1, S_kv)
            attn_mask = attn_mask.flatten(0, 1)

        needs_weights = len(self._attn_weight_hooks) > 0

        # Repeat keys and values if num_key_value_heads < num_heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # attn:         (N x H, S, V_h)
        # attn_weights: (N x H, S, S_kv)
        attn, attn_weights = self.sdpa(q, k, v, attn_mask, needs_weights)

        if attn_weights is not None:
            self._run_attn_weight_hooks(attn_weights)

        # (N x H, S, V_h) -> (N, H, S, V_h)
        attn = attn.unflatten(0, (-1, self.num_heads))

        # (N, H, S, V_h) -> (N, S, H, V_h)
        attn = attn.permute(0, 2, 1, 3)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attn = attn.flatten(-2, -1)

        # (N, S, V_proj) -> (N, S, M)
        attn = self.output_proj(attn)

        return attn  # type: ignore

    def _repeat_kv(self, seqs: Tensor) -> Tensor:
        """
        This is equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).
        The keys and values go from (N x H_kv, S_kv, head_dim) to
        (N x H, S_kv, head_dim)
        """
        if self.num_key_value_groups == 1:
            return seqs
        _, seq_len, head_dim = seqs.shape
        seqs = seqs[:, None, :, :].expand(
            -1, self.num_key_value_groups, seq_len, head_dim
        )
        return seqs.reshape(-1, seq_len, head_dim)

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()
        s += f", num_key_value_heads={self.num_key_value_heads}"
        return s
