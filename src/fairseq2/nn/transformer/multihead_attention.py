# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, MutableSequence, Optional, Protocol, Tuple, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle

from fairseq2.nn.incremental_state import IncrementalState, IncrementalStateBag
from fairseq2.nn.ops import repeat_interleave
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Linear, Projection
from fairseq2.nn.transformer.attention import SDPA, create_default_sdpa
from fairseq2.typing import DataType, Device, finaloverride, override


class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention layer."""

    num_heads: int
    model_dim: int

    _attn_weight_hooks: Dict[int, "AttentionWeightHook"]

    def __init__(self, model_dim: int, num_heads: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param num_heads:
            The number of attention heads.
        """
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self._attn_weight_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self,
        queries: Tensor,
        padding_mask: Optional[Tensor],
        keys: Tensor,
        values: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param queries:
            The queries. *Shape:* :math:`(N,S,M)`, where :math:`N` is the batch
            size, :math:`S` is the sequence length, and :math:`M` is the
            dimensionality of the model.
        :param padding_mask:
            The float padding mask of ``queries``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`V` is the value size.
        :param attn_mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(S,S_{kv})`, where
            :math:`S` is the sequence length and :math:`S_{kv}` is the key/value
            sequence length.
        :param key_padding_mask:
            The float padding mask indicating which key positions to ignore for
            the purpose of attention. *Shape:* :math:`(N,S_{kv})`, where
            :math:`N` is the batch size and :math:`S_{kv}` is the key/value
            sequence length.
        :param state_bag:
            The state bag to use for incremental evaluation.

        :returns:
            The attention values for ``queries``. *Shape:* :math:`(N,S,M)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`M` is the dimensionality of the model.
        """

    def register_attn_weight_hook(self, hook: "AttentionWeightHook") -> RemovableHandle:
        """Register an attention weight hook on the module.

        The hook will be called every time after the module computes attention
        weights.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._attn_weight_hooks)

        self._attn_weight_hooks[handle.id] = hook

        return handle

    def _run_attn_weight_hooks(self, attn: Tensor, attn_weights: Tensor) -> None:
        """Run registered attention weight hooks.

        :param attn:
            The computed attention values. *Shape:* :math:`(N,S,V)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`V` is the value size.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.

        :meta public:
        """
        for hook in self._attn_weight_hooks.values():
            hook(self, attn, attn_weights)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"num_heads={self.num_heads}, model_dim={self.model_dim}"


class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(
        self, m: MultiheadAttention, attn: Tensor, attn_weights: Tensor
    ) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attn:
            The computed attention values. *Shape:* :math:`(N,S,V)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`V` is the value size.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        """


class StoreAttentionWeights:
    """Stores attention weights in a provided storage.

    .. note::
        This class follows the :class:`AttentionWeightHook` protocol.
    """

    _storage: MutableSequence[Tuple[Tensor, Tensor]]

    def __init__(self, storage: MutableSequence[Tuple[Tensor, Tensor]]) -> None:
        """
        :param storage:
            The storage in which to store attention weights.
        """
        self._storage = storage

    def __call__(
        self, m: MultiheadAttention, attn: Tensor, attn_weights: Tensor
    ) -> None:
        self._storage.append((attn, attn_weights))


@final
class StandardMultiheadAttention(MultiheadAttention):
    """Represents a Transformer multi-head attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    num_key_value_heads: int
    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    pos_encoder: Optional[PositionEncoder]
    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]
    add_zero_attn: bool
    sdpa: SDPA
    head_scale_weight: Optional[Parameter]
    output_proj: Projection

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        num_key_value_heads: Optional[int] = None,
        q_proj: Optional[Projection] = None,
        k_proj: Optional[Projection] = None,
        v_proj: Optional[Projection] = None,
        pos_encoder: Optional[PositionEncoder] = None,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        sdpa: Optional[SDPA] = None,
        scale_heads: bool = False,
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
            The number of key/value heads for Grouped Query Attention as
            described in :cite:t:`https://doi.org/10.48550/arXiv.2305.13245`.
            If ``None`` or set to ``num_heads``, it is equivalent to standard
            Multi Head Attention (MHA); if set to 1, it is equivalent to Multi
            Query Attention (MQA).
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
        :param add_bias_kv:
            If ``True``, extends keys and values by a bias step.
        :param add_zero_attn:
            If ``True``, extends keys and values by an empty (i.e. zero) step.
        :param sdpa:
            The scaled dot-product attention module to compute head attentions.
            If ``None``, a default implementation will be used.
        :param scale_heads:
            If ``True``, applies head scaling as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`
        :param output_proj:
            The projection to produce final attentions. If ``None``, a default
            projection will be used.
        :param bias:
            If ``True``, query, key, value, and output projections learn an
            additive bias. Ignored for explicitly specified projections.
        """
        super().__init__(model_dim, num_heads)

        if num_key_value_heads is None:
            self.num_key_value_heads = num_heads
        else:
            if num_heads < num_key_value_heads:
                raise ValueError(
                    f"`num_heads` must be greater than or equal to `num_key_value_heads` ({num_key_value_heads}), but is {num_heads} instead."
                )

            if num_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"`num_heads` must be a multiple of `num_key_value_heads` ({num_key_value_heads}), but is {num_heads} instead."
                )

            self.num_key_value_heads = num_key_value_heads

        head_dim = model_dim // num_heads

        num_query_groups = num_heads // self.num_key_value_heads

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = QKVProjection(
                model_dim, model_dim, bias, device=device, dtype=dtype
            )
            k_proj = QKVProjection(
                model_dim,
                head_dim * self.num_key_value_heads,
                bias,
                device=device,
                dtype=dtype,
            )
            v_proj = QKVProjection(
                model_dim,
                head_dim * self.num_key_value_heads,
                bias,
                device=device,
                dtype=dtype,
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

            if (k_dim := k_proj.output_dim * num_query_groups) != q_proj.output_dim:
                raise ValueError(
                    f"`output_dim` of `q_proj` and `output_dim` of `k_proj` (times the number of query groups when GQA) must be equal, but are {q_proj.output_dim} and {k_dim} instead."
                )

            if k_proj.output_dim % self.num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `k_proj` must be a multiple of `num_key_value_heads` ({self.num_key_value_heads}), but is {k_proj.output_dim} instead."
                )

            if v_proj.output_dim % self.num_key_value_heads != 0:
                raise ValueError(
                    f"`output_dim` of `v_proj` must be a multiple of `num_key_value_heads` ({self.num_key_value_heads}), but is {v_proj.output_dim} instead."
                )

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        if pos_encoder is not None:
            if head_dim != pos_encoder.encoding_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` must be equal to the size of the header dimension ({head_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if add_bias_kv:
            bias_k_shp = (num_heads, 1, k_proj.output_dim // num_heads)
            bias_v_shp = (num_heads, 1, v_proj.output_dim // num_heads)

            # (H, 1, K_h)
            self.bias_k = Parameter(torch.empty(bias_k_shp, device=device, dtype=dtype))
            # (H, 1, V_h)
            self.bias_v = Parameter(torch.empty(bias_v_shp, device=device, dtype=dtype))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.add_zero_attn = add_zero_attn

        if sdpa is not None:
            self.sdpa = sdpa
        else:
            self.sdpa = create_default_sdpa()

        if scale_heads:
            self.head_scale_weight = Parameter(
                torch.empty(num_heads, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("head_scale_weight", None)

        v_dim = v_proj.output_dim * num_query_groups

        if output_proj is None:
            self.output_proj = AttentionOutputProjection(
                v_dim, model_dim, bias, device=device, dtype=dtype
            )
        else:
            if v_dim != output_proj.input_dim:
                raise ValueError(
                    f"`output_dim` of `v_proj` (times the number of query groups when GQA) and `input_dim` of `output_proj` must be equal, but are {v_dim} and {output_proj.input_dim} instead."
                )

            if output_proj.output_dim != model_dim:
                raise ValueError(
                    f"`output_dim` of `output_proj` must be equal to `model_dim` ({model_dim}), but is {output_proj.output_dim} instead."
                )

            self.output_proj = output_proj

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        if self.head_scale_weight is not None:
            nn.init.ones_(self.head_scale_weight)

    @finaloverride
    def forward(
        self,
        queries: Tensor,
        padding_mask: Optional[Tensor],
        keys: Tensor,
        values: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        # (N, S, M) -> (N, H, S, K_h)
        q = self._project_q(queries, padding_mask, state_bag)

        if self.training or state_bag is None:
            # k: (N, S_kv, M) -> (N, H_kv, S_kv, K_h)
            # v: (N, S_kv, M) -> (N, H_kv, S_kv, V_h)
            k, v = self._project_kv(keys, key_padding_mask, values)
        else:
            encoder_decoder_attn = keys is values and keys is not queries
            if encoder_decoder_attn:
                static_state = state_bag.get_state(self, StaticMultiheadAttentionState)

                # The K and V tensors of an encoder-decoder attention (i.e. the
                # projected encoder outputs) remain static during evaluation.
                if static_state is not None:
                    k = static_state.k
                    v = static_state.v
                else:
                    # k: (N, S_kv, M) -> (N, H_kv, S_kv, K_h)
                    # v: (N, S_kv, M) -> (N, H_kv, S_kv, V_h)
                    k, v = self._project_kv(keys, key_padding_mask, values)

                    state_bag.set_state(self, StaticMultiheadAttentionState(k, v))
            else:
                # k: (N, S_step, M) -> (N, H_kv, S_step, K_h)
                # v: (N, S_step, M) -> (N, H_kv, S_step, V_h)
                k, v = self._project_kv(keys, key_padding_mask, values, state_bag)

                state = state_bag.get_state(self, MultiheadAttentionState)
                if state is None:
                    state = MultiheadAttentionState(k, v)

                    state_bag.set_state(self, state)

                # k: (N, H_kv, S_kv, K_h)
                # v: (N, H_kv, S_kv, V_h)
                k, v, key_padding_mask = state.append(k, v, key_padding_mask)

        # With Grouped Query Attention, each key/value head is repeated.
        if (num_query_groups := self.num_heads // self.num_key_value_heads) > 1:
            # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, K_h)
            k = repeat_interleave(k, dim=1, repeat=num_query_groups)
            # (N, H_kv, S_kv, K_h) -> (N, H, S_kv, V_h)
            v = repeat_interleave(v, dim=1, repeat=num_query_groups)

        mask_pad = 0

        if self.bias_k is not None and self.bias_v is not None:
            batch_size = keys.size(0)

            # (H, 1, K_proj) -> (N, H, 1, K_proj)
            bias_k = self.bias_k.expand(batch_size, -1, -1, -1)
            # (H, 1, V_proj) -> (N x H, 1, V_proj)
            bias_v = self.bias_v.expand(batch_size, -1, -1, -1)

            # (N, H, S_kv, K_h) -> (N, H, S_kv + 1, K_h)
            k = torch.cat([k, bias_k], dim=2)
            # (N, H, S_kv, V_h) -> (N, H, S_kv + 1, V_h)
            v = torch.cat([v, bias_v], dim=2)

            mask_pad += 1

        if self.add_zero_attn:
            # (N, H, S_kv, K_h) -> (N, H, S_kv + 1, K_h)
            k = torch.cat([k, k.new_zeros((k.size(0), k.size(1), 1, k.size(3)))], dim=2)
            # (N, H, S_kv, V_h) -> (N, H, S_kv + 1, V_h)
            v = torch.cat([v, v.new_zeros((v.size(0), v.size(1), 1, v.size(3)))], dim=2)

            mask_pad += 1

        if mask_pad > 0:
            if attn_mask is not None:
                # (S, S_kv) -> (S, S_kv + mask_pad)
                attn_mask = pad(attn_mask, (0, mask_pad))

            if key_padding_mask is not None:
                # (N, S_kv) -> (N, S_kv + mask_pad)
                key_padding_mask = pad(key_padding_mask, (0, mask_pad))

        if key_padding_mask is not None:
            # (N, S_kv) -> (N, 1, 1, S_kv)
            key_padding_mask = key_padding_mask[:, None, None, :]

            # (N, 1, 1, S_kv) -> (N, H, 1, S_kv)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1)

            if attn_mask is None:
                # (N, H, 1, S_kv)
                attn_mask = key_padding_mask
            else:
                # (N, H, 1, S_kv) + ([H], S, S_kv) = (N, H, S, S_kv)
                attn_mask = key_padding_mask + attn_mask

        needs_weights = len(self._attn_weight_hooks) > 0

        # attn:         (N, H, S, V_h)
        # attn_weights: (N, H, S, S_kv)
        attn, attn_weights = self.sdpa(
            q, k, v, mask=attn_mask, needs_weights=needs_weights
        )

        if attn_weights is not None:
            self._run_attn_weight_hooks(attn, attn_weights)

        # (N, H, S, V_h) -> (N, S, H, V_h)
        attn = attn.transpose(1, 2)

        if self.head_scale_weight is not None:
            attn = torch.einsum("nshv,h->nshv", attn, self.head_scale_weight)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attn = attn.flatten(2, 3)

        # (N, S, V_proj) -> (N, S, M)
        attn = self.output_proj(attn)

        return attn  # type: ignore[no-any-return]

    def _project_q(
        self,
        queries: Tensor,
        padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, K_proj)
        q = self.q_proj(queries)

        # (N, S, K_proj) -> (N, H, S, K_h)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, padding_mask, state_bag=state_bag)

        return q  # type: ignore[no-any-return]

    def _project_kv(
        self,
        keys: Tensor,
        key_padding_mask: Optional[Tensor],
        values: Tensor,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Tensor]:
        # (N, S, K) -> (N, S, K_proj)
        k = self.k_proj(keys)
        # (N, S, V) -> (N, S, V_proj)
        v = self.v_proj(values)

        # (N, S, K_proj) -> (N, H, S, K_h)
        k = k.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)
        # (N, S, V_proj) -> (N, H, S, V_h)
        v = v.unflatten(-1, (self.num_key_value_heads, -1)).transpose(1, 2)

        if self.pos_encoder is not None:
            k = self.pos_encoder(k, key_padding_mask, state_bag=state_bag)

        return k, v

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.num_key_value_heads != self.num_heads:
            s = f"{s}, num_key_value_heads={self.num_key_value_heads}"

        if self.add_zero_attn:
            s = f"{s}, add_zero_attn=True"

        return s


class QKVProjection(Linear):
    """Represents the default projection used for queries, keys, and values."""

    def __init__(
        self,
        model_dim: int,
        output_dim: int,
        bias: bool,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(model_dim, output_dim, bias, device=device, dtype=dtype)

    @override
    def _do_reset_parameters(self) -> None:
        # Empirically observed the convergence to be much better with the scaled
        # initialization.
        nn.init.xavier_uniform_(self.weight, gain=2**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class AttentionOutputProjection(Linear):
    """Represents the default projection used for attention outputs."""

    def __init__(
        self,
        v_dim: int,
        model_dim: int,
        bias: bool,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(v_dim, model_dim, bias, device=device, dtype=dtype)

    @override
    def _do_reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiheadAttentionState(IncrementalState):
    """Holds the state of a :class:`MultiheadAttention` module during an
    incremental evaluation."""

    cache_reserve_size = 512
    """The reserved sequence length capacity of :attr:`k` and :attr:`v` will be
    increased by multiplies of the specified value."""

    seq_len: int
    """The current sequence length of :attr:`k` and :attr:`v`."""

    k: Tensor
    """The projected keys accumulated from the past incremental evaluation
    steps. *Shape:* :math:`(N,H,S,K_{proj})`, where :math:`N` is the batch
    size, :math:`H` is the number of heads, :math:`S` is the reserved sequence
    length capacity, and :math:`K_{proj}` is the projected key size."""

    v: Tensor
    """The projected values accumulated from the past incremental evaluation
    steps. *Shape:* :math:`(N,H,S,V_{proj})`, where :math:`N` is the batch
    size, :math:`H` is the number of heads, :math:`S` is the reserved sequence
    length capacity, and :math:`V_{proj}` is the projected value size."""

    key_padding_mask: Tensor
    """The float key padding mask accumulated from the past incremental
    evaluation steps. *Shape:* :math:`(N,S)`, where :math:`N` is the batch
    size and :math:`S` is the reserved sequence length capacity."""

    has_mask: bool

    def __init__(self, k: Tensor, v: Tensor) -> None:
        """
        :param k:
            The projected keys to bootstrap the internal state.
        :param v:
            The projected values to bootstrap the internal state.
        """
        batch_size, num_heads, _, head_dim = k.shape

        self.seq_len = 0

        self.k = k.new_empty((batch_size, num_heads, 0, head_dim))
        self.v = v.new_empty((batch_size, num_heads, 0, head_dim))

        self.key_padding_mask = k.new_zeros((batch_size, 0))

        self.has_mask = False

    def append(
        self, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Append the projected key, projected value, and float key padding mask
        of the current incremental evaluation step to :attr:`k`,
        :attr:`v`, and :attr:`key_padding_mask`.

        :param k:
            The projected key of the current incremental evaluation step.
            *Shape:* :math:`(N,H,S_{stp},K_{proj})`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{stp}` is the
            step length (e.g. 1), and :math:`K_{proj}` is the projected key
            size.
        :param v:
            The projected value of the current incremental evaluation step.
            *Shape:* :math:`(N,H,S_{stp},V_{proj})`, where :math:`N` is the
            batch size, :math:`H` is the number of heads, :math:`S_{stp}` is the
            step length (e.g. 1), and :math:`V_{proj}` is the projected value
            size.
        :param key_padding_mask:
            The float key padding mask of the current incremental evaluation
            step. *Shape:* :math:`(N,S_{stp})`, where :math:`N` is the batch
            size and :math:`S_{stp}` is the step length (e.g. 1).

        :returns:
            The projected keys, projected values, and float key padding mask
            that should be used to compute the attention.
        """
        seq_len = k.size(2)

        start, end = self.seq_len, self.seq_len + seq_len

        if end > self.k.size(2):
            batch_size, num_heads, seq_len, head_dim = k.shape

            # Ensure that the reserved space is always at least as long as the
            # input sequence.
            extra_capacity = self.cache_reserve_size * (
                (self.cache_reserve_size + seq_len - 1) // self.cache_reserve_size
            )

            cache_k = k.new_empty((batch_size, num_heads, extra_capacity, head_dim))
            cache_v = v.new_empty((batch_size, num_heads, extra_capacity, head_dim))

            self.k = torch.cat([self.k, cache_k], dim=2)
            self.v = torch.cat([self.v, cache_v], dim=2)

            cache_key_padding_mask = k.new_zeros((batch_size, extra_capacity))

            self.key_padding_mask = torch.cat(
                [self.key_padding_mask, cache_key_padding_mask], dim=1
            )

        self.k[:, :, start:end] = k
        self.v[:, :, start:end] = v

        if key_padding_mask is not None:
            self.has_mask = True

            self.key_padding_mask[:, start:end] = key_padding_mask

        self.seq_len = end

        k = self.k[:, :, :end]
        v = self.v[:, :, :end]

        key_padding_mask = self.key_padding_mask[:, :end] if self.has_mask else None

        return k, v, key_padding_mask

    @override
    def reorder(self, new_order: Tensor) -> None:
        self.k = self.k.index_select(0, new_order)
        self.v = self.v.index_select(0, new_order)

        self.key_padding_mask = self.key_padding_mask.index_select(0, new_order)


class StaticMultiheadAttentionState(IncrementalState):
    """Holds the state of an encoder-decoder :class:`MultiheadAttention` module
    during an incremental evaluation."""

    k: Tensor
    v: Tensor

    def __init__(self, k: Tensor, v: Tensor) -> None:
        """
        :param k:
            The encoder output projected as key.
        :param v:
            The encoder output projected as value.
        """
        self.k = k
        self.v = v

    @override
    def reorder(self, new_order: Tensor) -> None:
        self.k = self.k.index_select(0, new_order)
        self.v = self.v.index_select(0, new_order)
