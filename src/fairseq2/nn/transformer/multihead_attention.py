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

    def _run_attn_weight_hooks(self, attn_weights: Tensor) -> None:
        """Run registered attention weight hooks.

        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.

        :meta public:
        """
        for hook in self._attn_weight_hooks.values():
            hook(self, attn_weights)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"num_heads={self.num_heads}, model_dim={self.model_dim}"


class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(self, m: MultiheadAttention, attn_weights: Tensor) -> None:
        """
        :param m:
            The module that has computed the attention weights.
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

    _storage: MutableSequence[Tensor]

    def __init__(self, storage: MutableSequence[Tensor]) -> None:
        """
        :param storage:
            The storage in which to store attention weights.
        """
        self._storage = storage

    def __call__(self, m: "MultiheadAttention", attn_weights: Tensor) -> None:
        self._storage.append(attn_weights)


@final
class StandardMultiheadAttention(MultiheadAttention):
    """Represents a Transformer multi-head attention as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

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
            The projection to produce final attentions. If ``None``, a
            default projection will be used.
        :param bias:
            If ``True``, query, key, value, and output projections learn an
            additive bias. Ignored for explicitly specified projections.
        """
        super().__init__(model_dim, num_heads)

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = QKVProjection(model_dim, bias, device=device, dtype=dtype)
            k_proj = QKVProjection(model_dim, bias, device=device, dtype=dtype)
            v_proj = QKVProjection(model_dim, bias, device=device, dtype=dtype)
        else:
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError(
                    "`q_proj`, `k_proj`, and `v_proj` must be all specified."
                )

            if q_proj.input_dim != model_dim:
                raise ValueError(
                    f"`input_dim` of `q_proj` must be equal to `model_dim` ({model_dim}), but is {q_proj.input_dim} instead."
                )

            if q_proj.output_dim != k_proj.output_dim:
                raise ValueError(
                    f"`output_dim` of `q_proj` and `output_dim` of `k_proj` must be equal, but are {q_proj.output_dim} and {k_proj.output_dim} instead."
                )

        if k_proj.output_dim % num_heads != 0:
            raise ValueError(
                f"`output_dim` of `k_proj` must be divisible by `num_heads` ({num_heads}), but is {k_proj.output_dim} instead."
            )

        if v_proj.output_dim % num_heads != 0:
            raise ValueError(
                f"`output_dim` of `v_proj` must be divisible by `num_heads` ({num_heads}), but is {v_proj.output_dim} instead."
            )

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        if pos_encoder is not None:
            if (head_dim := k_proj.output_dim // num_heads) != pos_encoder.encoding_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and the size of the header key dimension must be equal, but are {pos_encoder.encoding_dim} and {head_dim} instead."
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

        if output_proj is None:
            self.output_proj = AttentionOutputProjection(
                v_proj.output_dim, model_dim, bias, device=device, dtype=dtype
            )
        else:
            if output_proj.input_dim != v_proj.output_dim:
                raise ValueError(
                    f"`input_dim` of `output_proj` and `output_dim` of `v_proj` must be equal, but are {output_proj.input_dim} and {v_proj.output_dim} instead."
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
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        # (*, M) -> (N, S, K_proj)
        q = self.q_proj(queries)

        if self.training or state_bag is None:
            # (*, K) -> (N, S_kv, K_proj)
            k = self.k_proj(keys)
            # (*, V) -> (N, S_kv, V_proj)
            v = self.v_proj(values)
        else:
            state = state_bag.get_state(self, MultiheadAttentionState)

            encoder_decoder_attn = keys is values and keys is not queries

            if encoder_decoder_attn:
                # The K and V tensors of an encoder-decoder attention (i.e. the
                # projected encoder outputs) remain static during an evaluation.
                if state is not None:
                    k = state.prev_k
                    v = state.prev_v
                else:
                    # (*, K) -> (N, S_kv, K_proj)
                    k = self.k_proj(keys)
                    # (*, V) -> (N, S_kv, V_proj)
                    v = self.v_proj(values)

                    state_bag.set_state(self, MultiheadAttentionState(k, v))
            else:
                # (*, K) -> (N, S_kv, K_proj)
                k = self.k_proj(keys)
                # (*, V) -> (N, S_kv, V_proj)
                v = self.v_proj(values)

                if state is not None:
                    k, v, key_padding_mask = state.append(k, v, key_padding_mask)
                else:
                    state_bag.set_state(
                        self, MultiheadAttentionState(k, v, key_padding_mask)
                    )

        # (N, S, K_proj) -> (N, S, H, K_h)
        q = q.unflatten(-1, (self.num_heads, -1))
        # (N, S_kv, K_proj) -> (N, S_kv, H, K_h)
        k = k.unflatten(-1, (self.num_heads, -1))
        # (N, S_kv, V_proj) -> (N, S_kv, H, V_h)
        v = v.unflatten(-1, (self.num_heads, -1))

        # (N, S, H, K_h) -> (N, H, S, K_h)
        q = q.transpose(1, 2)
        # (N, S_kv, H, K_h) -> (N, H, S_kv, K_h)
        k = k.transpose(1, 2)
        # (N, S_kv, H, V_h) -> (N, H, S_kv, V_h)
        v = v.transpose(1, 2)

        # (N, H, S, K_h) -> (N x H, S, K_h)
        q = q.flatten(0, 1)
        # (N, H, S_kv, K_h) -> (N x H, S_kv, K_h)
        k = k.flatten(0, 1)
        # (N, H, S_kv, V_h) -> (N x H, S_kv, V_h)
        v = v.flatten(0, 1)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, padding_mask, state_bag)
            k = self.pos_encoder(k, key_padding_mask)

        mask_pad = 0

        if self.bias_k is not None and self.bias_v is not None:
            batch_size = keys.size(0)

            # (H, 1, K_proj) -> (N x H, 1, K_proj)
            bias_k = self.bias_k.repeat(batch_size, 1, 1)
            # (H, 1, V_proj) -> (N x H, 1, V_proj)
            bias_v = self.bias_v.repeat(batch_size, 1, 1)

            # (N x H, S_kv, K_h) -> (N x H, S_kv + 1, K_h)
            k = torch.cat([k, bias_k], dim=1)
            # (N x H, S_kv, V_h) -> (N x H, S_kv + 1, V_h)
            v = torch.cat([v, bias_v], dim=1)

            mask_pad += 1

        if self.add_zero_attn:
            # (N x H, S_kv, K_h) -> (N x H, S_kv + 1, K_h)
            k = torch.cat([k, k.new_zeros((k.size(0), 1, k.size(2)))], dim=1)
            # (N x H, S_kv, V_h) -> (N x H, S_kv + 1, V_h)
            v = torch.cat([v, v.new_zeros((v.size(0), 1, v.size(2)))], dim=1)

            mask_pad += 1

        if mask_pad > 0:
            if attn_mask is not None:
                # (T, S_kv) -> (T, S_kv + mask_pad)
                attn_mask = pad(attn_mask, (0, mask_pad))

            if key_padding_mask is not None:
                # (N, S_kv) -> (N, S_kv + mask_pad)
                key_padding_mask = pad(key_padding_mask, (0, mask_pad))

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

        # attn:         (N x H, S, V_h)
        # attn_weights: (N x H, S, S_kv)
        attn, attn_weights = self.sdpa(q, k, v, attn_mask, needs_weights)

        if attn_weights is not None:
            self._run_attn_weight_hooks(attn_weights)

        # (N x H, S, V_h) -> (N, H, S, V_h)
        attn = attn.unflatten(0, (-1, self.num_heads))

        # (N, H, S, V_h) -> (N, S, H, V_h)
        attn = attn.permute(0, 2, 1, 3)

        if self.head_scale_weight is not None:
            attn = torch.einsum("nshv,h->nshv", attn, self.head_scale_weight)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attn = attn.flatten(-2, -1)

        # (N, S, V_proj) -> (N, S, M)
        attn = self.output_proj(attn)

        return attn  # type: ignore

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.add_zero_attn:
            s += ", add_zero_attn=True"

        return s


class QKVProjection(Linear):
    """Represents the default projection used for queries, keys, and values."""

    def __init__(
        self,
        model_dim: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(model_dim, model_dim, bias=bias, device=device, dtype=dtype)

    @override
    def _do_reset_parameters(self) -> None:
        # Empirically observed the convergence to be much better with the
        # scaled initialization.
        nn.init.xavier_uniform_(self.weight, gain=2**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class AttentionOutputProjection(Linear):
    """Represents the default projection used for attention outputs."""

    def __init__(
        self,
        v_proj_dim: int,
        model_dim: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(v_proj_dim, model_dim, bias=bias, device=device, dtype=dtype)

    @override
    def _do_reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiheadAttentionState(IncrementalState):
    """Holds the state of a :class:`MultiheadAttention` module during an
    incremental evaluation."""

    prev_k: Tensor
    """The projected keys accumulated from the past incremental evaluations.
    *Shape:* :math:`(N,S_{prv},K_{proj})`, where :math:`N` is the batch size,
    :math:`S_{prv}` is the accumulated key/value sequence length, and
    :math:`K_{proj}` is the projected key size."""

    prev_v: Tensor
    """The projected values accumulated from the past incremental evaluations.
    *Shape:* :math:`(N,S_{prv},V_{proj})`, where :math:`N` is the batch size,
    :math:`S_{prv}` is the accumulated key/value sequence length, and
    :math:`V_{proj}` is the projected value size."""

    prev_key_padding_mask: Optional[Tensor]
    """The float key padding mask accumulated from the past incremental
    evaluations. *Shape:* :math:`(N,S_{prv})`, where :math:`N` is the batch size
    and :math:`S_{prv}` is the accumulated key/value sequence length."""

    def __init__(
        self, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> None:
        """
        :param k:
            The initial projected keys. *Shape:* :math:`(N,S_{int},K_{proj})`,
            where :math:`N` is the batch size, :math:`S_{int}` is the initial
            key/value sequence length, and :math:`K_{proj}` is the projected key
            size.
        :param v:
            The initial projected values. *Shape:* :math:`(N,S_{int},V_{proj})`,
            where :math:`N` is the batch size, :math:`S_{int}` is the initial
            key/value sequence length, and :math:`V_{proj}` is the projected
            value size.
        :param key_padding_mask:
            The initial float key padding mask. *Shape:* :math:`(N,S_{int})`,
            where :math:`N` is the batch size and :math:`S_{int}` is the initial
            key/value sequence length.
        """
        self.prev_k = k
        self.prev_v = v

        self.prev_key_padding_mask = key_padding_mask

    def append(
        self, k: Tensor, v: Tensor, key_padding_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Append the projected key, projected value, and float key padding mask
        of the current incremental evaluation to :attr:`prev_k`, :attr:`prev_v`,
        and :attr:`key_padding_mask`.

        :param k:
            The projected key of the current incremental evaluation. *Shape:*
            :math:`(N,S_{stp},K_{proj})`, where :math:`N` is the batch size,
            :math:`S_{stp}` is the step length (e.g. 1), and :math:`K_{proj}` is
            the projected key size.
        :param v:
            The projected value of the current incremental evaluation. *Shape:*
            :math:`(N,S_{stp},V_{proj})`, where :math:`N` is the batch size,
            :math:`S_{stp}` is the step length (e.g. 1), and :math:`V_{proj}` is
            the projected value size.
        :param key_padding_mask:
            The float key padding mask of the current incremental evaluation.
            *Shape:* :math:`(N,S_{stp})`, where :math:`N` is the batch size and
            :math:`S_{stp}` is the step length (e.g. 1).

        :returns:
            The projected keys, projected values, and float key padding mask
            that should be used to compute the attention.
        """
        seq_len = k.size(1)

        prev_seq_len = self.prev_k.size(1)

        self.prev_k = torch.cat([self.prev_k, k], dim=1)
        self.prev_v = torch.cat([self.prev_v, v], dim=1)

        # Appending the key padding mask is trickier since the previous or
        # current mask can be `None`.
        self._append_key_padding_mask(key_padding_mask, seq_len, prev_seq_len)

        return self.prev_k, self.prev_v, self.prev_key_padding_mask

    def _append_key_padding_mask(
        self, curr_mask: Optional[Tensor], curr_seq_len: int, prev_seq_len: int
    ) -> None:
        prev_mask = self.prev_key_padding_mask

        if prev_mask is None and curr_mask is None:
            return

        batch_size = self.prev_k.size(0)

        # One of the masks can be `None`. We have to ensure that both of them
        # are fully materialized before concatenating.
        if prev_mask is None:
            prev_mask = self.prev_k.new_zeros((batch_size, prev_seq_len))

        if curr_mask is None:
            curr_mask = self.prev_k.new_zeros((batch_size, curr_seq_len))

        self.prev_key_padding_mask = torch.cat([prev_mask, curr_mask], dim=1)

    @override
    def reorder(self, new_order: Tensor) -> None:
        self.prev_k = self.prev_k.index_select(0, new_order)
        self.prev_v = self.prev_v.index_select(0, new_order)

        if self.prev_key_padding_mask is not None:
            mask = self.prev_key_padding_mask.index_select(0, new_order)

            self.prev_key_padding_mask = mask
