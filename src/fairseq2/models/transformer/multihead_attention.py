# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, MutableSequence
from typing import TYPE_CHECKING, Final, Protocol, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from fairseq2.data_type import DataType
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.models.transformer.attention_bias import AttentionBiasCache
from fairseq2.models.transformer.sdpa.base import SDPA
from fairseq2.nn import (
    BatchLayout,
    ColumnShardedLinear,
    IncrementalState,
    IncrementalStateBag,
    LayerNorm,
    Linear,
    PositionEncoder,
    Projection,
    RowShardedLinear,
)
from fairseq2.nn.utils.module import get_name_or_self
from fairseq2.ops import repeat_interleave
from fairseq2.utils.warn import _warn_deprecated


class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention layer."""

    def __init__(self) -> None:
        super().__init__()

        self._attn_weight_hooks: dict[int, AttentionWeightHook] = OrderedDict()

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        keys: Tensor,
        keys_layout: BatchLayout,
        values: Tensor,
        bias_cache: AttentionBiasCache,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to query. *Shape:* :math:`(N,S,M)`, where :math:`N` is
            the batch size, :math:`S` is the sequence length, and :math:`M` is
            the dimensionality of the model.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`V` is the value size.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            The attention values for ``seqs``. *Shape:* :math:`(N,S,M)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`M` is the dimensionality of the model.
        """

    if TYPE_CHECKING:
        __call__ = forward

    def register_attn_weight_hook(self, hook: AttentionWeightHook) -> RemovableHandle:
        """Register an attention weight hook on the module.

        The hook will be called every time after the module has computed
        attention weights.

        :param hook:
            The hook to register.

        :returns:
            A handle that can be used to remove the added hook by calling
            ``handle.remove()``.
        """
        handle = RemovableHandle(self._attn_weight_hooks)

        self._attn_weight_hooks[handle.id] = hook

        return handle


class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(
        self, m: MultiheadAttention, attns: Tensor, attn_weights: Tensor
    ) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attns:
            The computed attention values. *Shape:* :math:`(N,S,V)`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`V` is the value size.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        """


@final
class AttentionWeightStoreHook(AttentionWeightHook):
    """Stores attention weights in a provided storage.

    .. note::
        This class follows the :class:`AttentionWeightHook` protocol.
    """

    def __init__(self, storage: MutableSequence[tuple[Tensor, Tensor]]) -> None:
        """
        :param storage: The storage in which to store attention weights.
        """
        self._storage = storage

    def __call__(
        self, m: MultiheadAttention, attns: Tensor, attn_weights: Tensor
    ) -> None:
        self._storage.append((attns, attn_weights))


@final
class StandardMultiheadAttention(MultiheadAttention):
    """Represents a Transformer multi-head attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

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
        state_factory: AttentionStateFactory | None = None,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
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
        :param kv_dim:
            The dimensionality of keys and values. May be useful for encoder-
            decoder attention. If ``None``, ``model_dim`` will be used.
        :param q_proj:
            The projection to apply to sequences before computing attention. If
            ``None``, a default projection will be used.
        :param k_proj:
            The projection to apply to keys before computing attention. If
            ``None``, a default projection will be used.
        :param v_proj:
            The projection to apply to values before computing attention. If
            ``None``, a default projection will be used.
        :param qkv_proj_init_fn:
            The callable to initialize the q, k, v projections.
        :param q_norm:
            If ``True``, applies Layer Normalization to projected queries.
        :param k_norm:
            If ``True``, applies Layer Normalization to projected keys.
        :param pos_encoder:
            The position encoder to apply to sequences and keys after projection.
        :param sdpa:
            The :class:`SDPA` module to compute head attentions. If ``None``, a
            default implementation will be used.
        :param output_proj:
            The projection to produce final attentions. If ``None``, a default
            projection will be used.
        :param output_proj_init_fn:
            The callable to initialize the output projection.
        :param bias:
            If ``True``, query, key, and value projections learn an additive
            bias. Ignored for explicitly specified projections.
        :param output_proj_bias:
            If ``True``, output projection learns an additive bias. If ``None``,
            the value of ``bias`` is used. Ignored for explicitly specified
            projections.
        :param state_factory:
            The factory to construct :class:`AttentionState` instances for
            incremental decoding.
        """
        super().__init__()

        self.num_heads = num_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_heads
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

        if head_dim is None:
            head_dim = model_dim // num_heads

        self.head_dim = head_dim

        if kv_dim is None:
            kv_dim = model_dim

        self.kv_dim = kv_dim

        self.num_query_groups = num_heads // num_key_value_heads

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = Linear(
                model_dim,
                head_dim * num_heads,
                bias,
                init_fn=qkv_proj_init_fn or init_qkv_projection,
                device=device,
                dtype=dtype,
            )
            k_proj = Linear(
                kv_dim,
                head_dim * num_key_value_heads,
                bias,
                init_fn=qkv_proj_init_fn or init_qkv_projection,
                device=device,
                dtype=dtype,
            )
            v_proj = Linear(
                kv_dim,
                head_dim * num_key_value_heads,
                bias,
                init_fn=qkv_proj_init_fn or init_qkv_projection,
                device=device,
                dtype=dtype,
            )
        else:
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError("`q_proj`, `k_proj`, `v_proj` must be all specified.")

            _warn_deprecated(
                "`q_proj`, `k_proj`, and `v_proj` parameters of `StandardMultiheadAttention` are deprecated and will be removed in fairseq2 v0.12."
            )

            if qkv_proj_init_fn is not None:
                raise ValueError(
                    "`qkv_proj_init_fn` must not be specified when `q_proj`, `k_proj`, `v_proj` are specified."
                )

            if q_proj.input_dim != kv_dim:
                raise ValueError(
                    f"`q_proj.input_dim` must be equal to `kv_dim` ({kv_dim}), but is {q_proj.input_dim} instead."
                )

            k_dim = k_proj.output_dim * self.num_query_groups
            if k_dim != q_proj.output_dim:
                raise ValueError(
                    f"`q_proj.output_dim` and `k_proj.output_dim` (or times the number of query groups when GQA) must be equal, but they are {q_proj.output_dim} and {k_dim} instead."
                )

            if k_proj.output_dim % num_key_value_heads != 0:
                raise ValueError(
                    f"`k_proj.output_dim` must be a multiple of `num_key_value_heads` ({num_key_value_heads}), but is {k_proj.output_dim} instead."
                )

            if v_proj.output_dim % num_key_value_heads != 0:
                raise ValueError(
                    f"`v_proj.output_dim` must be a multiple of `num_key_value_heads` ({num_key_value_heads}), but is {v_proj.output_dim} instead."
                )

        self.q_proj: Projection
        self.k_proj: Projection
        self.v_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.q_proj = q_proj
            self.k_proj = k_proj
            self.v_proj = v_proj
        else:
            if isinstance(q_proj, Linear):
                self.q_proj = ColumnShardedLinear.from_linear(
                    q_proj, gangs.tp, gather_output=False
                )
            else:
                self.q_proj = q_proj

            del q_proj

            if isinstance(k_proj, Linear):
                self.k_proj = ColumnShardedLinear.from_linear(
                    k_proj, gangs.tp, gather_output=False
                )
            else:
                self.k_proj = k_proj

            del k_proj

            if isinstance(v_proj, Linear):
                self.v_proj = ColumnShardedLinear.from_linear(
                    v_proj, gangs.tp, gather_output=False
                )
            else:
                self.v_proj = v_proj

            del v_proj

        self.q_norm: LayerNorm | None
        self.k_norm: LayerNorm | None

        self.register_module("q_norm", q_norm)
        self.register_module("k_norm", k_norm)

        if pos_encoder is not None:
            if head_dim != pos_encoder.encoding_dim:
                raise ValueError(
                    f"`pos_encoder.encoding_dim` must be equal to the size of the head dimension ({head_dim}), but is {pos_encoder.encoding_dim} instead."
                )

            pos_encoder = pos_encoder
        else:
            pos_encoder = None

        self.pos_encoder: PositionEncoder | None

        self.register_module("pos_encoder", pos_encoder)

        self.sdpa = sdpa

        v_dim = self.v_proj.output_dim * self.num_query_groups

        if output_proj is None:
            if output_proj_bias is None:
                output_proj_bias = bias

            output_proj = Linear(
                v_dim,
                model_dim,
                output_proj_bias,
                init_fn=output_proj_init_fn or init_mha_output_projection,
                device=device,
                dtype=dtype,
            )
        else:
            _warn_deprecated(
                "`output_proj` parameter of `StandardMultiheadAttention` is deprecated and will be removed in fairseq2 v0.12."
            )

            if output_proj_init_fn is not None:
                raise ValueError(
                    "`output_proj_init_fn` must not be specified when `output_proj` is specified."
                )

            if v_dim != output_proj.input_dim:
                raise ValueError(
                    f"`v_proj.output_dim` (or times the number of query groups when GQA) and `output_proj.input_dim` must be equal, but they are {v_dim} and {output_proj.input_dim} instead."
                )

            if output_proj.output_dim != model_dim:
                raise ValueError(
                    f"`output_proj.output_dim` must be equal to `model_dim` ({model_dim}), but is {output_proj.output_dim} instead."
                )

        self.output_proj: Projection

        if gangs is None or gangs.tp.size == 1:
            self.output_proj = output_proj
        else:
            if isinstance(output_proj, Linear):
                self.output_proj = RowShardedLinear.from_linear(
                    output_proj, gangs.tp, scatter_input=False
                )
            else:
                self.ouput_proj = output_proj

            del output_proj

        self.state_factory = state_factory

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
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, H, K_h)
        q = self._project_q(seqs, seqs_layout, state_bag)

        if self.training or state_bag is None:
            # k: (N, S_kv, M) -> (N, S_kv, H_kv, K_h)
            # v: (N, S_kv, M) -> (N, S_kv, H_kv, V_h)
            k, v = self._project_kv(keys, keys_layout, values)
        else:
            if seqs is keys:  # self attention
                if keys_layout.packed:
                    raise ValueError("`keys` must not be a packed batch.")

                if keys_layout.padded:
                    raise ValueError("`keys` must not be a padded batch.")

                # k: (N, S_step, M) -> (N, S_step, H_kv, K_h)
                # v: (N, S_step, M) -> (N, S_step, H_kv, V_h)
                k, v = self._project_kv(keys, keys_layout, values, state_bag)

                state = state_bag.maybe_get_state(self, AttentionState)
                if state is None:
                    state_factory = self.state_factory or FullAttentionState

                    state = state_factory(
                        k, v, state_bag.max_num_steps, state_bag.capacity_increment
                    )

                    state_bag.set_state(self, state)
                else:
                    state.append(k, v)

                    # k: (N, S_kv, H_kv, K_h)
                    # v: (N, S_kv, H_kv, V_h)
                    k, v = state.get()

                    keys_layout = BatchLayout.of(k)
            else:
                state = state_bag.maybe_get_state(self, AttentionState)
                if state is None:
                    # k: (N, S_kv, M) -> (N, S_kv, H_kv, K_h)
                    # v: (N, S_kv, M) -> (N, S_kv, H_kv, V_h)
                    k, v = self._project_kv(keys, keys_layout, values)

                    state_factory = self.state_factory or StaticAttentionState

                    state = state_factory(
                        k, v, max_seq_len=k.size(1), capacity_increment=None
                    )

                    state_bag.set_state(self, state)
                else:
                    # k: (N, S_kv, H_kv, K_h)
                    # v: (N, S_kv, H_kv, V_h)
                    k, v = state.get()

        # With Grouped Query Attention, each key/value head is repeated.
        if self.num_query_groups > 1:
            # (N, S_kv, H_kv, K_h) -> (N, S_kv, H, K_h)
            k = repeat_interleave(k, dim=-2, repeat=self.num_query_groups)
            # (N, S_kv, H_kv, K_h) -> (N, S_kv, H, V_h)
            v = repeat_interleave(v, dim=-2, repeat=self.num_query_groups)

        needs_weights = len(self._attn_weight_hooks) > 0

        # attns:        (N, S, H, V_h)
        # attn_weights: (N, H, S, S_kv)
        attns, attn_weights = self.sdpa(
            q, seqs_layout, k, keys_layout, v, bias_cache, needs_weights=needs_weights
        )

        del q
        del k
        del v

        if attn_weights is not None:
            for hook in self._attn_weight_hooks.values():
                hook(self, attns, attn_weights)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attns = attns.flatten(-2, -1)

        # (N, S, V_proj) -> (N, S, M)
        return self.output_proj(attns)

    def _project_q(
        self,
        seqs: Tensor,
        seqs_layout: BatchLayout,
        state_bag: IncrementalStateBag | None = None,
    ) -> Tensor:
        # (N, S, M) -> (N, S, K_proj)
        q = self.q_proj(seqs)

        # (N, S, K_proj) -> (N, S, H, K_h)
        q = q.unflatten(-1, (-1, self.head_dim))

        if self.q_norm is not None:
            q = self.q_norm(q)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, seqs_layout, state_bag=state_bag)

        return q

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

        # (N, S, K_proj) -> (N, S, H, K_h)
        k = k.unflatten(-1, (-1, self.head_dim))
        # (N, S, V_proj) -> (N, S, H, V_h)
        v = v.unflatten(-1, (-1, self.head_dim))

        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.pos_encoder is not None:
            k = self.pos_encoder(k, keys_layout, state_bag=state_bag)

        return k, v

    @override
    def extra_repr(self) -> str:
        """:meta private:"""
        s = f"num_heads={self.num_heads}"

        if self.num_key_value_heads != self.num_heads:
            s = f"{s}, num_key_value_heads={self.num_key_value_heads}"

        if self.num_query_groups > 1:
            s = f"{s}, num_query_groups={self.num_query_groups}"

        if self.state_factory is not None:
            state_factory = get_name_or_self(self.state_factory)

            s = f"{s}, state_factory={state_factory}"

        return s


def init_qkv_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention input projection."""
    # Empirically observed the convergence to be much better with the scaled
    # initialization.
    nn.init.xavier_uniform_(proj.weight, gain=2**-0.5)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


def init_mha_output_projection(proj: Linear) -> None:
    """Initialize ``proj`` as a multi-head attention output projection."""
    nn.init.xavier_uniform_(proj.weight)

    if proj.bias is not None:
        nn.init.zeros_(proj.bias)


class AttentionState(IncrementalState):
    """Holds the projected keys and values of a :class:`MultiheadAttention`
    module during incremental decoding."""

    @abstractmethod
    def append(self, k: Tensor, v: Tensor) -> None:
        """Update the state with ``k`` and ``v``.

        :param k:
            The projected keys of the current step. *Shape:*
            :math:`(N,1,H,K_{proj})`, where :math:`N` is the batch size,
            :math:`1` is the step length, :math:`H` is the number of heads, and
            :math:`K_{proj}` is the projected key size.
        :param v:
            The projected values of the current step. *Shape:*
            :math:`(N,1,H,V_{proj})`, where :math:`N` is the batch size,
            :math:`1` is the step length, :math:`H` is the number of heads, and
            :math:`V_{proj}` is the projected value size.
        """

    @abstractmethod
    def get(self) -> tuple[Tensor, Tensor]:
        """Return the state that should be used to compute the attention.

        :returns:
            - The projected keys.
            - The projected values.
        """


class AttentionStateFactory(Protocol):
    """Constructs instances of :class:`AttentionState`."""

    def __call__(
        self, k: Tensor, v: Tensor, max_seq_len: int, capacity_increment: int | None
    ) -> AttentionState:
        """
        :param k:
            The initial projected keys.
        :param v:
            The initial projected values.
        :param max_seq_len:
            The maximum sequence length.
        :param capacity_increment:
            The sequence length capacity of state tensors will be incremented by
            multiples of this value. If ``None``, state tensors will be
            preallocated with a capacity of ``max_seq_len``.
        """


@final
class FullAttentionState(AttentionState):
    """Holds the past projected keys and values of a :class:`MultiheadAttention`
    module during incremental decoding."""

    _seq_len: int
    """The current sequence length of :attr:`k` and :attr:`v`."""

    _k: Tensor
    """The projected keys accumulated from the past decoding steps. *Shape:*
    :math:`(N,S_{rsv},H,K_{proj})`, where :math:`N` is the batch size, :math:`H`
    is the number of heads, :math:`S_{rsv}` is the reserved sequence length
    capacity, and :math:`K_{proj}` is the projected key size."""

    _v: Tensor
    """The projected values accumulated from the past decoding steps. *Shape:*
    :math:`(N,S_{rsv},H,V_{proj})`, where :math:`N` is the batch size, :math:`H`
    is the number of heads, :math:`S_{rsv}` is the reserved sequence length
    capacity, and :math:`V_{proj}` is the projected value size."""

    _capacity_increment: int | None
    """The sequence length capacity of :attr:`k` and :attr:`v` is incremented by
    multiples of this value."""

    def __init__(
        self, k: Tensor, v: Tensor, max_seq_len: int, capacity_increment: int | None
    ) -> None:
        if capacity_increment is not None and capacity_increment < 1:
            raise ValueError(
                f"`capacity_increment` must be greater than or equal to 1, but is {capacity_increment} instead."
            )

        batch_size, seq_len, num_heads, head_dim = k.shape

        init_capacity = 0 if capacity_increment else max_seq_len

        self._k = k.new_empty((batch_size, init_capacity, num_heads, head_dim))
        self._v = v.new_empty((batch_size, init_capacity, num_heads, head_dim))

        self._seq_len = 0

        self._capacity_increment = capacity_increment

        self._expand_kv(seq_len)

        self._k[:, :seq_len] = k
        self._v[:, :seq_len] = v

        self._seq_len = seq_len

    @override
    def append(self, k: Tensor, v: Tensor) -> None:
        input_seq_len = k.size(1)

        self._expand_kv(input_seq_len)

        pos = self._seq_len

        self._k[:, pos : pos + input_seq_len] = k
        self._v[:, pos : pos + input_seq_len] = v

        self._seq_len += input_seq_len

    def _expand_kv(self, input_seq_len: int) -> None:
        if self._capacity_increment is None:
            return

        batch_size, capacity, num_heads, head_dim = self._k.shape

        new_seq_len = self._seq_len + input_seq_len

        if new_seq_len <= capacity:
            return

        inc = self._capacity_increment

        capacity = ((new_seq_len + inc - 1) // inc) * inc

        k = self._k.new_empty((batch_size, capacity, num_heads, head_dim))
        v = self._v.new_empty((batch_size, capacity, num_heads, head_dim))

        if self._seq_len > 0:
            k[:, : self._seq_len] = self._k[:, : self._seq_len]
            v[:, : self._seq_len] = self._v[:, : self._seq_len]

        self._k = k
        self._v = v

    @override
    def get(self) -> tuple[Tensor, Tensor]:
        k = self._k[:, : self._seq_len]
        v = self._v[:, : self._seq_len]

        return k, v

    @override
    def reorder(self, new_order: Tensor) -> None:
        self._k = self._k.index_select(0, new_order)
        self._v = self._v.index_select(0, new_order)

    @override
    def size_bytes(self) -> int:
        batch_size, _, num_heads, head_dim = self._k.shape

        numel = 2 * batch_size * self._seq_len * num_heads * head_dim

        return numel * self._k.dtype.itemsize

    @override
    def capacity_bytes(self) -> int:
        return 2 * self._k.numel() * self._k.dtype.itemsize


@final
class LocalAttentionState(AttentionState):
    """Holds the past :attr:`attn_window_len` projected keys and values of a
    :class:`MultiheadAttention` module during incremental decoding.

    The intended use of this class is with Sliding Window Attention as described
    in :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`.
    """

    _seq_len: int
    """The current sequence length of :attr:`k` and :attr:`v`."""

    _attn_window_len: int
    """The attention window length."""

    _k: Tensor
    """The projected keys accumulated from the past :attr:`attn_window_len`
    decoding steps. *Shape:* :math:`(N,S_{rsv},H,K_{proj})`, where :math:`N` is
    the batch size, :math:`H` is the number of heads, :math:`S_{rsv}` is the
    reserved sequence length capacity, and :math:`K_{proj}` is the projected key
    size."""

    _v: Tensor
    """The projected values accumulated from the past :attr:`attn_window_len`
    decoding steps. *Shape:* :math:`(N,S_{rsv},H,V_{proj})`, where :math:`N` is
    the batch size, :math:`H` is the number of heads, :math:`S_{rsv}` is the
    reserved sequence length capacity, and :math:`V_{proj}` is the projected
    value size."""

    _capacity_increment: int | None
    """The sequence length capacity of :attr:`k` and :attr:`v` is incremented by
    multiples of this value."""

    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        max_seq_len: int,
        attn_window_len: int,
        capacity_increment: int | None,
    ) -> None:
        if capacity_increment is not None and capacity_increment < 1:
            raise ValueError(
                f"`capacity_increment` must be greater than or equal to 1, but is {capacity_increment} instead."
            )

        self._attn_window_len = min(max_seq_len, attn_window_len)

        batch_size, seq_len, num_heads, head_dim = k.shape

        init_capacity = 0 if capacity_increment else self._attn_window_len

        self._k = k.new_empty((batch_size, init_capacity, num_heads, head_dim))
        self._v = v.new_empty((batch_size, init_capacity, num_heads, head_dim))

        self._seq_len = 0

        self._capacity_increment = capacity_increment

        self._expand_kv(seq_len)

        copy_len = min(seq_len, self._attn_window_len)

        self._k[:, :copy_len] = k[:, -copy_len:]
        self._v[:, :copy_len] = v[:, -copy_len:]

        self._seq_len = seq_len

    @override
    def append(self, k: Tensor, v: Tensor) -> None:
        input_seq_len = k.size(1)

        self._expand_kv(input_seq_len)

        if self._seq_len + input_seq_len > self._attn_window_len:
            copy_len = min(input_seq_len, self._attn_window_len)

            if copy_len == self._attn_window_len:
                pos = 0
            else:
                self._k = torch.roll(self._k, shifts=-copy_len, dims=1)
                self._v = torch.roll(self._v, shifts=-copy_len, dims=1)

                pos = self._attn_window_len - copy_len
        else:
            pos = self._seq_len

            copy_len = input_seq_len

        self._k[:, pos : pos + copy_len] = k[:, -copy_len:]
        self._v[:, pos : pos + copy_len] = v[:, -copy_len:]

        self._seq_len += input_seq_len

    def _expand_kv(self, input_seq_len: int) -> None:
        if self._capacity_increment is None:
            return

        batch_size, capacity, num_heads, head_dim = self._k.shape

        new_seq_len = self._seq_len + input_seq_len

        if new_seq_len <= capacity or capacity == self._attn_window_len:
            return

        inc = self._capacity_increment

        capacity = min(((new_seq_len + inc - 1) // inc) * inc, self._attn_window_len)

        k = self._k.new_empty((batch_size, capacity, num_heads, head_dim))
        v = self._v.new_empty((batch_size, capacity, num_heads, head_dim))

        if self._seq_len > 0:
            k[:, : self._seq_len] = self._k[:, : self._seq_len]
            v[:, : self._seq_len] = self._v[:, : self._seq_len]

        self._k = k
        self._v = v

    @override
    def get(self) -> tuple[Tensor, Tensor]:
        k = self._k[:, : self._seq_len]
        v = self._v[:, : self._seq_len]

        return k, v

    @override
    def reorder(self, new_order: Tensor) -> None:
        self._k = self._k.index_select(0, new_order)
        self._v = self._v.index_select(0, new_order)

    @override
    def size_bytes(self) -> int:
        if self._seq_len >= self._attn_window_len:
            return self.capacity_bytes()

        batch_size, _, num_heads, head_dim = self._k.shape

        numel = 2 * batch_size * self._seq_len * num_heads * head_dim

        return numel * self._k.dtype.itemsize

    @override
    def capacity_bytes(self) -> int:
        return 2 * self._k.numel() * self._k.dtype.itemsize


@final
class LocalAttentionStateFactory(AttentionStateFactory):
    """Constructs instances of :class:`LocalAttentionState`."""

    def __init__(self, attn_window_len: int) -> None:
        """
        :param attn_window_len:
            The attention window length.
        """
        self.attn_window_len: Final = attn_window_len

    def __call__(
        self,
        k: Tensor,
        v: Tensor,
        max_seq_len: int,
        capacity_increment: int | None,
    ) -> LocalAttentionState:
        return LocalAttentionState(
            k, v, max_seq_len, self.attn_window_len, capacity_increment
        )

    def __repr__(self) -> str:
        return f"LocalAttentionStateFactory(attn_window_len={self.attn_window_len})"


@final
class StaticAttentionState(AttentionState):
    """Holds the static projected keys and values (e.g. encoder-decoder) of a
    :class:`MultiheadAttention` module during incremental decoding."""

    def __init__(
        self, k: Tensor, v: Tensor, max_seq_len: int, capacity_increment: int | None
    ) -> None:
        self._k = k
        self._v = v

    @override
    def append(self, k: Tensor, v: Tensor) -> None:
        raise NotSupportedError(
            f"`{StaticAttentionState}` does not support `append()`."
        )

    @override
    def get(self) -> tuple[Tensor, Tensor]:
        return self._k, self._v

    @override
    def reorder(self, new_order: Tensor) -> None:
        if new_order.size(0) != self._k.size(0):
            self._k = self._k.index_select(0, new_order)
            self._v = self._v.index_select(0, new_order)

    @override
    def size_bytes(self) -> int:
        return self.capacity_bytes()

    @override
    def capacity_bytes(self) -> int:
        return 2 * self._k.numel() * self._k.dtype.itemsize
