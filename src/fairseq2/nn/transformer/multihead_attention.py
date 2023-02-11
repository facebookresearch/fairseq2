# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, MutableSequence, Optional, Protocol, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride, override
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle

from fairseq2.nn.incremental_state import IncrementalState, IncrementalStateBag
from fairseq2.nn.positional_embedding import PositionalEmbedding
from fairseq2.nn.projection import Projection, ResettableProjection
from fairseq2.nn.transformer.attention import (
    AttentionFunction,
    default_scaled_dot_product_attention,
)
from fairseq2.nn.utils import to_float_mask
from fairseq2.typing import DataType, Device


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
        self, k: Tensor, v: Tensor, padding_mask: Optional[Tensor] = None
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
        """Appends the projected key, the projected value, and the float key
        padding mask of the current step to :attr:`prev_k`, :attr:`prev_v`, and
        :attr:`padding_mask`.

        :param k:
            The projected key of the current step. *Shape:*
            :math:`(N,S_{stp},K_{proj})`, where :math:`N` is the batch size,
            :math:`S_{stp}` is the step length (e.g. 1), and :math:`K_{proj}` is
            the projected key size.
        :param v:
            The projected value of the current step. *Shape:*
            :math:`(N,S_{stp},V_{proj})`, where :math:`N` is the batch size,
            :math:`S_{stp}` is the step length (e.g. 1), and :math:`V_{proj}` is
            the projected value size.
        :param padding_mask:
            The float key padding mask of the current step. *Shape:*
            :math:`(N,S_{stp})`, where :math:`N` is the batch size and
            :math:`S_{stp}` is the step length (e.g. 1).

        :returns:
            The projected keys, the projected values, and the key padding mask
            that should be used by the current step to compute the attentions.
        """
        seq_len = k.size(1)

        prev_seq_len = self.prev_k.size(1)

        self.prev_k = torch.cat([self.prev_k, k], dim=1)
        self.prev_v = torch.cat([self.prev_v, v], dim=1)

        # Appending the key padding mask is trickier than K and V since either
        # the previous or the current mask can be None.
        self._append_padding_mask(padding_mask, seq_len, prev_seq_len)

        return self.prev_k, self.prev_v, self.prev_padding_mask

    def _append_padding_mask(
        self, curr_mask: Optional[Tensor], curr_seq_len: int, prev_seq_len: int
    ) -> None:
        prev_mask = self.prev_padding_mask

        if prev_mask is None and curr_mask is None:
            return

        bsz = self.prev_k.size(0)

        # One of the masks can still be None. We have to ensure that both of
        # them are fully materialized before concatenating them.
        if prev_mask is None:
            prev_mask = self.prev_k.new_zeros((bsz, prev_seq_len))

        if curr_mask is None:
            curr_mask = self.prev_k.new_zeros((bsz, curr_seq_len))

        self.prev_padding_mask = torch.cat([prev_mask, curr_mask], dim=1)

    @override
    def reorder(self, new_order: Tensor) -> None:
        self.prev_k = self.prev_k.index_select(0, new_order)
        self.prev_v = self.prev_v.index_select(0, new_order)

        if self.prev_padding_mask is not None:
            self.prev_padding_mask = self.prev_padding_mask.index_select(0, new_order)


class AttentionWeightHook(Protocol):
    """Represents a hook to pass to
    :meth:`~MultiheadAttention.register_attn_weight_hook`."""

    def __call__(self, m: "MultiheadAttention", attn_weights: Tensor) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,T,S)`, where
            :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`S` is the source sequence length.
        """


class StoreAttentionWeights:
    """Stores the attention weights in a given list.

    The user of this class is responsible for clearing the list, or popping the
    attention weights.

    .. note::
        This class follows the :class:`AttentionWeightHook` protocol.
    """

    def __init__(self, attn_weights: MutableSequence[Tensor]) -> None:
        """
        :param attn_weights:
            The list in which to store the attention weights.
        """
        self._attn_weights = attn_weights

    def __call__(self, m: "MultiheadAttention", attn_weights: Tensor) -> None:
        """
        :param m:
            The module that has computed the attention weights.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,T,S)`, where
            :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`S` is the source sequence length.
        """
        self._attn_weights.append(attn_weights)


class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention."""

    num_heads: int
    """The number of attention heads."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    _attn_weight_hooks: Dict[int, AttentionWeightHook]

    def __init__(self, num_heads: int, model_dim: int) -> None:
        """
        :param num_heads:
            The number of attention heads.
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        """
        super().__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim

        self._attn_weight_hooks = OrderedDict()

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param x:
            The input to query for. *Shape:* :math:`(N,T,M)`, or :math:`(T,M)`
            when unbatched, where :math:`N` is the batch size, :math:`T` is the
            target sequence length, and :math:`M` is the model size.
        :param keys:
            The keys. *Shape:* :math:`(N,S,K)`, or :math:`(S,K)` when unbatched,
            where :math:`N` is the batch size, :math:`S` is the source sequence
            length, and :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S,V)`, or :math:`(S,V)` when
            unbatched, where :math:`N` is the batch size, :math:`S` is the
            source sequence length, and :math:`V` is the value size.
        :param attn_mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(T,S)`, where :math:`T` is
            the target sequence length and :math:`S` is the source sequence
            length.
        :param padding_mask:
            The boolean or float key padding mask indicating which key positions
            to ignore for the purpose of attention. *Shape:* :math:`(N,S)`, or
            :math:`(S)` when unbatched, where :math:`N` is the batch size and
            :math:`S` is the source sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The attentions. *Shape:* :math:`(N,T,M)`, or :math:`(T,M)` when
            unbatched, where :math:`N` is the batch size, :math:`T` is the
            target sequence length, and :math:`M` is the model size.

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
    def __init__(
        self, model_dim: int, device: Optional[Device], dtype: Optional[DataType]
    ) -> None:
        super().__init__(model_dim, model_dim, bias=True, device=device, dtype=dtype)

    @override
    def reset_parameters(self) -> None:
        # Empirically observed the convergence to be much better with the
        # scaled initialization.
        nn.init.xavier_uniform_(self.weight, gain=2**-0.5)

        if self.bias is not None:
            nn.init.zeros_(self.bias)


class InternalOutProjection(ResettableProjection):
    def __init__(
        self,
        v_proj_dim: int,
        model_dim: int,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> None:
        super().__init__(v_proj_dim, model_dim, bias=True, device=device, dtype=dtype)

    @override
    def reset_parameters(self) -> None:
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
    pos_embed: Optional[PositionalEmbedding]
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
        pos_embed: Optional[PositionalEmbedding] = None,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        attn_fn: Optional[AttentionFunction] = None,
        attn_dropout_p: float = 0.0,
        out_proj: Optional[Projection] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param num_heads:
            The number of attention heads.
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs); must be
            specified if ``q_proj`` is ``None``; otherwise, will be inferred
            from ``q_proj``.
        :param q_proj:
            The projection to apply to inputs before computing attention. If
            ``None``, a default projection will be used.
        :param k_proj:
            The projection to apply to keys before computing attention. If
            ``None``, a default projection will be used.
        :param v_proj:
            The projection to apply to values before computing attention. If
            ``None``, a default projection will be used.
        :param pos_embed:
            The positional embedding to add to inputs and keys after applying
            projection.
        :param add_bias_kv:
            If ``True``, extends keys and values by a bias step.
        :param add_zero_attn:
            If ``True``, extends keys and values by an empty (i.e. zero) step.
        :param attn_fn:
            The function to compute head attentions. If ``None``, a default
            implementation of the scaled dot-product attention will be used.
        :param attn_dropout_p:
            The dropout probability on attention weights.
        :param out_proj:
            The projection to produce final attentions. If ``None``, a
            default projection will be used.
        """
        fct_kwargs: Dict[str, Any] = {"device": device, "dtype": dtype}

        if model_dim is None:
            if q_proj is not None:
                model_dim = q_proj.inp_dim
            else:
                raise ValueError("`model_dim` must be specified.")

        super().__init__(num_heads, model_dim)

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

        if pos_embed is not None:
            if (head_dim := k_proj.out_dim // num_heads) != pos_embed.embedding_dim:
                raise ValueError(
                    f"`embedding_dim` of `pos_embed` ({pos_embed.embedding_dim}) does not match the size of the per-header key dimension ({head_dim})."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if add_bias_kv:
            # (H, 1, K_h)
            self.bias_k = Parameter(
                torch.empty((num_heads, 1, k_proj.out_dim // num_heads), **fct_kwargs)
            )
            # (H, 1, V_h)
            self.bias_v = Parameter(
                torch.empty((num_heads, 1, v_proj.out_dim // num_heads), **fct_kwargs)
            )
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.add_zero_attn = add_zero_attn

        if attn_fn is None:
            self.attn_fn = default_scaled_dot_product_attention
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

    @finaloverride
    def forward(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        padding_mask = self._prepare_padding_mask(keys, padding_mask)

        # (*, M) -> (N, T, K_proj)
        q = self._forward_proj(self.q_proj, x)

        if self.training or state_bag is None:
            # (*, K) -> (N, S, K_proj)
            k = self._forward_proj(self.k_proj, keys)
            # (*, V) -> (N, S, V_proj)
            v = self._forward_proj(self.v_proj, values)
        else:
            state = state_bag.get_state(self, MultiheadAttentionState)

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

                    state_bag.set_state(self, MultiheadAttentionState(k, v))
            else:
                # (*, K) -> (N, S, K_proj)
                k = self._forward_proj(self.k_proj, keys)
                # (*, V) -> (N, S, V_proj)
                v = self._forward_proj(self.v_proj, values)

                if state is not None:
                    k, v, padding_mask = state.append(k, v, padding_mask)
                else:
                    state_bag.set_state(
                        self, MultiheadAttentionState(k, v, padding_mask)
                    )

        # (N, T, K_proj) -> (N, T, H, K_h)
        q = q.unflatten(-1, (self.num_heads, -1))
        # (N, S, K_proj) -> (N, S, H, K_h)
        k = k.unflatten(-1, (self.num_heads, -1))
        # (N, S, V_proj) -> (N, S, H, V_h)
        v = v.unflatten(-1, (self.num_heads, -1))

        # (N, T, H, K_h) -> (N, H, T, K_h)
        q = q.transpose(1, 2)
        # (N, S, H, K_h) -> (N, H, S, K_h)
        k = k.transpose(1, 2)
        # (N, S, H, V_h) -> (N, H, S, V_h)
        v = v.transpose(1, 2)

        # (N, H, T, K_h) -> (N x H, T, K_h)
        q = q.flatten(0, 1)
        # (N, H, S, K_h) -> (N x H, S, K_h)
        k = k.flatten(0, 1)
        # (N, H, S, V_h) -> (N x H, S, V_h)
        v = v.flatten(0, 1)

        if self.pos_embed is not None:
            q = self.pos_embed(q, state_bag)
            k = self.pos_embed(k)

        mask_pad = 0

        if self.bias_k is not None and self.bias_v is not None:
            bsz = keys.size(0)

            # (H, 1, K_proj) -> (N x H, 1, K_proj)
            bias_k = self.bias_k.repeat(bsz, 1, 1)
            # (H, 1, V_proj) -> (N x H, 1, V_proj)
            bias_v = self.bias_v.repeat(bsz, 1, 1)

            # (N x H, S, K_h) -> (N x H, S + 1, K_h)
            k = torch.cat([k, bias_k], dim=1)
            # (N x H, S, V_h) -> (N x H, S + 1, V_h)
            v = torch.cat([v, bias_v], dim=1)

            mask_pad += 1

        if self.add_zero_attn:
            # (N x H, S, K_h) -> (N x H, S + 1, K_h)
            k = torch.cat([k, k.new_zeros((k.size(0), 1, k.size(2)))], dim=1)
            # (N x H, S, V_h) -> (N x H, S + 1, V_h)
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

        needs_weights = len(self._attn_weight_hooks) > 0

        # attn:         (N x H, T, V_h)
        # attn_weights: (N x H, T, S)
        attn, attn_weights = self.attn_fn(
            q, k, v, attn_mask, self.attn_dropout_p, needs_weights, self.training
        )

        if attn_weights is not None:
            self._run_attn_weight_hooks(attn_weights)

        # (N x H, T, V_h) -> (N, H, T, V_h)
        attn = attn.unflatten(0, (-1, self.num_heads))

        # (N, H, T, V_h) -> (N, T, H, V_h)
        attn = attn.permute(0, 2, 1, 3)

        # (N, T, H, V_h) -> (N, T, V_proj)
        attn = attn.flatten(-2, -1)

        # (N, T, V_proj) -> (N, T, M)
        attn = self.out_proj(attn)

        if x.dim() == 3:
            return attn
        else:
            # (1, T, M) -> (T, M)
            return attn.squeeze(0)

    def _forward_proj(self, fn: Projection, x: Tensor) -> Tensor:
        x = fn(x)

        if x.dim() == 3:
            # (N, S, X_proj)
            return x
        else:
            # (S, X_proj) -> (1, S, X_proj)
            return x.unsqueeze(0)

    def _prepare_padding_mask(
        self, keys: Tensor, mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        if mask is not None:
            return to_float_mask(mask, dtype=keys.dtype)
        else:
            return mask
