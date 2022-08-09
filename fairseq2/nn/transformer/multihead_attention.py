# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter
from torch.utils.hooks import RemovableHandle

from ..incremental_state import IncrementalState, IncrementalStateBag
from ..projection import Projection, ResettableProjection
from ..utils import to_float_mask
from .attention import AttentionFunction, scaled_dot_product_attention


class MultiheadAttentionState(IncrementalState):
    """Holds the state of a :class:`MultiheadAttention` module during an
    incremental evaluation."""

    prev_k: Tensor
    """The projected keys accumulated from the previous steps. *Shape:*
    :math:`(N,S_{prv},K_{proj})`, where :math:`N` is the batch size,
    :math:`S_{prv}` is the source sequence length of the previous step, and
    :math:`K_{proj}` is the projected key size."""

    prev_v: Tensor
    """The projected values accumulated from the previous steps. *Shape:*
    :math:`(N,S_{prv},V_{proj})`, where :math:`N` is the batch size,
    :math:`S_{prv}` is the source sequence length of the previous step, and
    :math:`V_{proj}` is the projected value size."""

    prev_padding_mask: Optional[Tensor]
    """The float key padding mask accumulated from the previous steps. *Shape:*
    :math:`(N,S_{prv})`, where :math:`N` is the batch size and :math:`S_{prv}`
    is the source sequence length of the previous step."""

    def __init__(
        self,
        k: Tensor,
        v: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> None:
        """
        :param k:
            The initial projected keys. *Shape:* :math:`(N,S_{int},K_{proj})`,
            where :math:`N` is the batch size, :math:`S_{int}` is the initial
            source sequence length, and :math:`K_{proj}` is the projected key
            size.
        :param v:
            The initial projected values. *Shape:* :math:`(N,S_{int},V_{proj})`,
            where :math:`N` is the batch size, :math:`S_{int}` is the initial
            source sequence length, and :math:`V_{proj}` is the projected value
            size.
        :param padding_mask:
            The initial float key padding mask. *Shape:* :math:`(N,S_{int})`,
            where :math:`N` is the batch size and :math:`S_{int}` is the initial
            source sequence length.
        """
        self.prev_k = k
        self.prev_v = v

        self.prev_padding_mask = padding_mask

    def append(
        self, k: Tensor, v: Tensor, padding_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Appends the projected key, value and the float key padding mask of
        the current step to :attr:`prev_k`, :attr:`prev_v`, and
        :attr:`padding_mask`.

        :param k:
            The projected key of the current step. *Shape:*
            :math:`(N,S_{stp},K_{proj})`, where :math:`N` is the batch size,
            :math:`S_{stp}` is the step length (i.e. 1), and :math:`K_{proj}` is
            the projected key size.
        :param v:
            The projected value of the current step. *Shape:*
            :math:`(N,S_{stp},V_{proj})`, where :math:`N` is the batch size,
            :math:`S_{stp}` is the step length (i.e. 1), and :math:`V_{proj}` is
            the projected value size.
        :param padding_mask:
            The float key padding mask of the current step. *Shape:*
            :math:`(N,S_{stp})`, where :math:`N` is the batch size and
            :math:`S_{stp}` is the step length (i.e. 1).

        :returns:
            The projected keys, values and the key padding mask that should be
            used by the current step to compute the attentions.
        """
        seq_len = k.size(1)

        prev_seq_len = self.prev_k.size(1)

        self.prev_k = torch.cat([self.prev_k, k], dim=1)
        self.prev_v = torch.cat([self.prev_v, v], dim=1)

        # Appending the key padding mask is trickier than K and V since either
        # the previous or the current mask can be `None`.
        self._append_padding_mask(padding_mask, seq_len, prev_seq_len)

        return self.prev_k, self.prev_v, self.prev_padding_mask

    def _append_padding_mask(
        self, curr_mask: Optional[Tensor], curr_seq_len: int, prev_seq_len: int
    ) -> None:
        prev_mask = self.prev_padding_mask

        if prev_mask is None and curr_mask is None:
            return

        bsz = self.prev_k.size(0)

        # One of the masks can still be `None`. We have to ensure that both of
        # them are fully materialized before concatenating them.
        if prev_mask is None:
            prev_mask = self.prev_k.new_zeros((bsz, prev_seq_len))

        if curr_mask is None:
            curr_mask = self.prev_k.new_zeros((bsz, curr_seq_len))

        self.prev_padding_mask = torch.cat([prev_mask, curr_mask], dim=1)

    def reorder(self, new_order: Tensor) -> None:  # override
        self.prev_k = self.prev_k.index_select(0, new_order)
        self.prev_v = self.prev_v.index_select(0, new_order)

        if self.prev_padding_mask is not None:
            self.prev_padding_mask = self.prev_padding_mask.index_select(0, new_order)


class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(self, m: MultiheadAttention, attn_weights: Tensor) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,T,S)`, where
            :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`S` is the source sequence length.
        """


class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention."""

    num_heads: int
    """The number of attention heads."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    _attn_weight_hooks: Dict[int, AttentionWeightHook]

    def __init__(self, num_heads: int, model_dim: int, batch_first: bool) -> None:
        """
        :param num_heads:
            The number of attention heads.
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        """
        super().__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim

        self.batch_first = batch_first

        self._attn_weight_hooks = {}

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        incremental_state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param x:
            The input to query for. *Shape:* :math:`(T,M)` when unbatched,
            :math:`(N,T,M)` when :attr:`batch_first` is ``True``, or
            :math:`(T,N,M)` when :attr:`batch_first` is ``False``,
            where :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`M` is the model size.
        :param keys:
            The keys. *Shape:* :math:`(S,K)` when unbatched, :math:`(N,S,K)`
            when :attr:`batch_first` is ``True``, or :math:`(S,N,K)` when
            :attr:`batch_first` is ``False``, where :math:`N` is the batch size,
            :math:`S` is the source sequence length, and :math:`K` is the key
            size.
        :param values:
            The values. *Shape:* :math:`(S,V)` when unbatched, :math:`(N,S,V)`
            when :attr:`batch_first` is ``True``, or :math:`(S,N,V)` when
            :attr:`batch_first` is ``False``, where :math:`N` is the batch size,
            :math:`S` is the source sequence length, and :math:`V` is the value
            size.
        :param attn_mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(T,S)`, where :math:`T` is
            the target sequence length and :math:`S` is the source sequence
            length.
        :param padding_mask:
            The boolean or float key padding mask indicating which key positions
            to ignore for the purpose of attention. *Shape:* :math:`(S)` when
            unbatched, :math:`(N,S)` when :attr:`batch_first` is ``True``, or
            :math:`(S,N)` when :attr:`batch_first` is ``False``, where :math:`N`
            is the batch size and :math:`S` is the source sequence length.
        :param incremental_state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The attentions. *Shape:* :math:`(T,M)` when unbatched,
            :math:`(N,T,M)` when :attr:`batch_first` is ``True``, or
            :math:`(T,N,M)` when :attr:`batch_first` is ``False``, where
            :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`M` is the model size.

        .. note::
            For a boolean key padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend. For a float key
            padding mask, the mask values will be added to the attention
            weights.
        """

    def register_attn_weight_hook(self, hook: AttentionWeightHook) -> RemovableHandle:
        """Registers an attention weight hook on the module.

        The hook will be called every time after :meth:`forward` has computed
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

    def _run_attn_weight_hooks(self, attn_weights: Tensor) -> None:
        """Runs registered attention weight hooks.

        A :class:`MultiheadAttention` implementation should call this method
        after computing attention weights in its :meth:`forward` method.

        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,T,S)`, where
            :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`S` is the source sequence length.

        :meta public:
        """
        for hook in self._attn_weight_hooks.values():
            hook(self, attn_weights)

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"num_heads={self.num_heads}, model_dim={self.model_dim}"


class InternalQKVProjection(ResettableProjection):
    def __init__(self, model_dim: int, device, dtype) -> None:
        super().__init__(model_dim, model_dim, bias=True, device=device, dtype=dtype)

    def reset_parameters(self) -> None:  # override
        # Empirically observed the convergence to be much better with the
        # scaled initialization.
        nn.init.xavier_uniform_(self.weight, gain=2**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class InternalOutProjection(ResettableProjection):
    def __init__(self, v_proj_dim: int, model_dim: int, device, dtype) -> None:
        super().__init__(v_proj_dim, model_dim, bias=True, device=device, dtype=dtype)

    def reset_parameters(self) -> None:  # override
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


@final
class StandardMultiheadAttention(MultiheadAttention):
    """Represents a Transformer multi-head attention as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`."""

    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]
    add_zero_attn: bool
    attn_fn: AttentionFunction
    attn_dropout_p: float
    out_proj: Projection

    def __init__(
        self,
        num_heads: int,
        model_dim: Optional[int] = None,
        q_proj: Optional[Projection] = None,
        k_proj: Optional[Projection] = None,
        v_proj: Optional[Projection] = None,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        attn_fn: Optional[AttentionFunction] = None,
        attn_dropout_p: float = 0.0,
        out_proj: Optional[Projection] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """
        :param num_heads:
            The number of attention heads.
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs); must be
            specified if ``q_proj`` is ``None``; otherwise, will be inferred
            from ``q_proj``.
        :param q_proj:
            The projection to apply to provided inputs before computing
            attention. If ``None``, a default projection will be used.
        :param k_proj:
            The projection to apply to provided keys before computing attention.
            If ``None``, a default projection will be used.
        :param v_proj:
            The projection to apply to provided values before computing
            attention. If ``None``, a default projection will be used.
        :param add_bias_kv:
            If ``True``, extends provided keys and values by a bias step.
        :param add_zero_attn:
            If ``True``, extends provided keys and values by an empty (i.e.
            zero) step.
        :param attn_fn:
            The function to compute head attentions. If ``None``, a default
            implementation of the scaled dot-product attention will be used.
        :param attn_dropout_p:
            The dropout probability on attention weights.
        :param out_proj:
            The projection to produce final attentions. If ``None``, a
            default projection will be used.
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        """
        fct_kwargs: Dict = {"device": device, "dtype": dtype}

        if model_dim is None:
            if q_proj is not None:
                model_dim = q_proj.inp_dim
            else:
                raise ValueError("`model_dim` must be specified.")

        super().__init__(num_heads, model_dim, batch_first)

        # TODO: Scale heads

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = InternalQKVProjection(model_dim, **fct_kwargs)
            k_proj = InternalQKVProjection(model_dim, **fct_kwargs)
            v_proj = InternalQKVProjection(model_dim, **fct_kwargs)
        else:
            if q_proj is None or k_proj is None or v_proj is None:
                raise ValueError(
                    "`q_proj`, `k_proj`, and `v_proj` must be all specified."
                )

            if q_proj.inp_dim != model_dim:
                raise ValueError(
                    f"`inp_dim` of `q_proj` ({q_proj.inp_dim}) does not match `model_dim` ({model_dim})."
                )

            if q_proj.out_dim != k_proj.out_dim:
                raise ValueError(
                    f"`out_dim` of `q_proj` ({q_proj.out_dim}) does not match `out_dim` of `k_proj` ({k_proj.out_dim})."
                )

        if k_proj.out_dim % num_heads != 0:
            raise ValueError(
                f"`out_dim` of `k_proj` ({k_proj.out_dim}) is not divisible by `num_heads` ({num_heads})."
            )

        if v_proj.out_dim % num_heads != 0:
            raise ValueError(
                f"`out_dim` of `v_proj` ({v_proj.out_dim}) is not divisible by `num_heads` ({num_heads})."
            )

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, k_proj.out_dim), **fct_kwargs))
            self.bias_v = Parameter(torch.empty((1, v_proj.out_dim), **fct_kwargs))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.add_zero_attn = add_zero_attn

        if attn_fn is None:
            self.attn_fn = scaled_dot_product_attention
        else:
            self.attn_fn = attn_fn

        self.attn_dropout_p = attn_dropout_p

        if out_proj is None:
            self.out_proj = InternalOutProjection(
                v_proj.out_dim, model_dim, **fct_kwargs
            )
        else:
            if out_proj.inp_dim != v_proj.out_dim:
                raise ValueError(
                    f"`inp_dim` of `out_proj` ({out_proj.inp_dim}) does not match `out_dim` of `v_proj` ({v_proj.out_dim})."
                )

            if out_proj.out_dim != model_dim:
                raise ValueError(
                    f"`out_dim` of `out_proj` ({out_proj.out_dim}) does not match `model_dim` ({model_dim})."
                )

            self.out_proj = out_proj

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        incremental_state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:  # override
        padding_mask = self._prepare_padding_mask(padding_mask)

        # (*, M) -> (N, T, K_proj)
        q = self._forward_proj(self.q_proj, x)

        if self.training or incremental_state_bag is None:
            # (*, K) -> (N, S, K_proj)
            k = self._forward_proj(self.k_proj, keys)
            # (*, V) -> (N, S, V_proj)
            v = self._forward_proj(self.v_proj, values)
        else:
            state = incremental_state_bag.get_state(self, MultiheadAttentionState)

            enc_dec_attn = keys is values and keys is not x

            if enc_dec_attn:
                # The K and V tensors of an encoder-decoder attention (i.e. the
                # projected encoder outputs) remain static during an evaluation.
                if state is not None:
                    k = state.prev_k
                    v = state.prev_v
                else:
                    # (*, K) -> (N, S, K_proj)
                    k = self._forward_proj(self.k_proj, keys)
                    # (*, V) -> (N, S, V_proj)
                    v = self._forward_proj(self.v_proj, values)

                    incremental_state_bag.set_state(self, MultiheadAttentionState(k, v))
            else:
                # (*, K) -> (N, S, K_proj)
                k = self._forward_proj(self.k_proj, keys)
                # (*, V) -> (N, S, V_proj)
                v = self._forward_proj(self.v_proj, values)

                if state is not None:
                    k, v, padding_mask = state.append(k, v, padding_mask)
                else:
                    incremental_state_bag.set_state(
                        self, MultiheadAttentionState(k, v, padding_mask)
                    )

        mask_pad = 0

        if self.bias_k is not None and self.bias_v is not None:
            # (N, S, K_proj) -> (N, S + 1, K_proj)
            k = torch.cat([k, self.bias_k.expand(k.size(0), 1, k.size(2))], dim=1)
            # (N, S, V_proj) -> (N, S + 1, V_proj)
            v = torch.cat([v, self.bias_v.expand(v.size(0), 1, v.size(2))], dim=1)

            mask_pad += 1

        if self.add_zero_attn:
            # (N, S, K_proj) -> (N, S + 1, K_proj)
            k = torch.cat([k, k.new_zeros((k.size(0), 1, k.size(2)))], dim=1)
            # (N, S, V_proj) -> (N, S + 1, V_proj)
            v = torch.cat([v, v.new_zeros((v.size(0), 1, v.size(2)))], dim=1)

            mask_pad += 1

        if mask_pad > 0:
            if attn_mask is not None:
                # (T, S) -> (T, S + mask_pad)
                attn_mask = F.pad(attn_mask, (0, mask_pad))

            if padding_mask is not None:
                # (N, S) -> (N, S + mask_pad)
                padding_mask = F.pad(padding_mask, (0, mask_pad))

        if padding_mask is not None:
            #       (N, S) -> (N, 1, 1, S)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            # (N, 1, 1, S) -> (N, H, 1, S)
            padding_mask = padding_mask.expand(-1, self.num_heads, -1, -1)
            # (N, H, 1, S) -> (N x H, 1, S)
            padding_mask = padding_mask.flatten(0, 1)

            if attn_mask is None:
                # (N x H, 1, S)
                attn_mask = padding_mask
            else:
                # (N x H, 1, S) + (T, S) = (N x H, T, S)
                attn_mask = padding_mask + attn_mask

        # (N, T, K_proj) -> (N, T, H, K_h)
        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))
        # (N, S, V_proj) -> (N, S, H, V_h)
        v = v.unflatten(-1, (self.num_heads, -1))

        # (N, T, H, K_h) -> (N, H, T, K_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # (N, S, H, V_h) -> (N, H, S, V_h)
        v = v.transpose(1, 2)

        # (N, H, T, K_h) -> (N x H, T, K_h)
        q = q.flatten(0, 1)
        k = k.flatten(0, 1)
        # (N, H, S, V_h) -> (N x H, S, V_h)
        v = v.flatten(0, 1)

        # attn:         (N x H, T, V_h)
        # attn_weights: (N x H, T, S)
        attn, attn_weights = self.attn_fn(
            q, k, v, attn_mask, self.attn_dropout_p, self.training
        )

        self._run_attn_weight_hooks(attn_weights)

        # (N x H, T, V_h) -> (N, H, T, V_h)
        attn = attn.unflatten(0, (-1, self.num_heads))

        if self.batch_first:
            # (N, H, T, V_h) -> (N, T, H, V_h)
            attn = attn.permute(0, 2, 1, 3)
        else:
            # (N, H, T, V_h) -> (T, N, H, V_h)
            attn = attn.permute(2, 0, 1, 3)

        # (*, H, V_h) -> (*, V_proj)
        attn = attn.flatten(-2, -1)

        # (*, V_proj) -> (*, M)
        attn = self.out_proj(attn)

        if x.dim() == 2:
            if self.batch_first:
                # (1, T, M) -> (T, M)
                return attn.squeeze(0)
            else:
                # (T, 1, M) -> (T, M)
                return attn.squeeze(1)

        return attn

    def _forward_proj(self, fn: Projection, x: Tensor) -> Tensor:
        x = fn(x)

        if x.dim() == 3:
            if not self.batch_first:
                # (S, N, X_proj) -> (N, S, X_proj)
                return x.transpose(0, 1)
            else:
                # (N, S, X_proj)
                return x
        else:
            # (S, X_proj) -> (1, S, X_proj)
            return x.unsqueeze(0)

    def _prepare_padding_mask(self, mask: Optional[Tensor]) -> Optional[Tensor]:
        if mask is not None:
            mask = to_float_mask(mask, dtype=self.k_proj.weight.dtype)

            if not self.batch_first:
                # (S, N) -> (N, S)
                return mask.transpose(0, 1)

        return mask
