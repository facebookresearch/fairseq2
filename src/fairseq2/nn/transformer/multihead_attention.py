# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, MutableSequence, Optional, Protocol, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from overrides import override
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
from fairseq2.nn.utils.fn import get_name
from fairseq2.nn.utils.mask import to_float_mask


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

    prev_padding_mask: Optional[Tensor]
    """The float key padding mask accumulated from the past incremental
    evaluations. *Shape:* :math:`(N,S_{prv})`, where :math:`N` is the batch size
    and :math:`S_{prv}` is the accumulated key/value sequence length."""

    def __init__(
        self, k: Tensor, v: Tensor, padding_mask: Optional[Tensor] = None
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
        :param padding_mask:
            The initial float key padding mask. *Shape:* :math:`(N,S_{int})`,
            where :math:`N` is the batch size and :math:`S_{int}` is the initial
            key/value sequence length.
        """
        self.prev_k = k
        self.prev_v = v

        self.prev_padding_mask = padding_mask

    def append(
        self, k: Tensor, v: Tensor, padding_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Append the projected key, projected value, and float key padding mask
        of the current incremental evaluation to :attr:`prev_k`, :attr:`prev_v`,
        and :attr:`padding_mask`.

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
        :param padding_mask:
            The float key padding mask of the current incremental evaluation.
            *Shape:* :math:`(N,S_{stp})`, where :math:`N` is the batch size and
            :math:`S_{stp}` is the step length (e.g. 1).

        :returns:
            The projected keys, projected values, and key padding mask that
            should be used to compute the attention.
        """
        seq_len = k.size(1)

        prev_seq_len = self.prev_k.size(1)

        self.prev_k = torch.cat([self.prev_k, k], dim=1)
        self.prev_v = torch.cat([self.prev_v, v], dim=1)

        # Appending the key padding mask is trickier than K and V since the
        # previous or current mask can be None.
        self._append_padding_mask(padding_mask, seq_len, prev_seq_len)

        return self.prev_k, self.prev_v, self.prev_padding_mask

    def _append_padding_mask(
        self, curr_mask: Optional[Tensor], curr_seq_len: int, prev_seq_len: int
    ) -> None:
        prev_mask = self.prev_padding_mask

        if prev_mask is None and curr_mask is None:
            return

        bsz = self.prev_k.size(0)

        # One of the masks can be None. We have to ensure that both of them are
        # fully materialized before concatenating them.
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
        """
        :param m:
            The module that has computed the attention weights.
        :param attn_weights:
            The computed attention weights. *Shape:* :math:`(N,S,S_{kv})`, where
            :math:`N` is the batch size, :math:`S` is the sequence length, and
            :math:`S_{kv}` is the key/value sequence length.
        """
        self._storage.append(attn_weights)


class MultiheadAttention(Module, ABC):
    """Represents a Transformer multi-head attention."""

    num_heads: int
    model_dim: int

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
            The input to query. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is
            the sequence length, and :math:`M` is the model size.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, or :math:`(S_{kv},K)` when
            unbatched, where :math:`N` is the batch size, :math:`S_{kv}` is the
            key/value sequence length, and :math:`K` is the key size.
        :param values:
            The values. *Shape:* :math:`(N,S_{kv},V)`, or :math:`(S_{kv},V)`
            when unbatched, where :math:`N` is the batch size, :math:`S_{kv}` is
            the key/value sequence length, and :math:`V` is the value size.
        :param attn_mask:
            The float mask that will be added to the attention weights before
            computing the attention. *Shape:* :math:`(S,S_{kv})`, where
            :math:`S` is the sequence length and :math:`S_{kv}` is the key/value
            sequence length.
        :param padding_mask:
            The boolean or float padding mask indicating which key positions to
            ignore for the purpose of attention. *Shape:* :math:`(N,S_{kv})`, or
            :math:`(S_{kv})` when unbatched, where :math:`N` is the batch size
            and :math:`S_{kv}` is the key/value sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The attention values. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the model size.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend. For a float
            padding mask, the mask values will be added to the attention
            weights.
        """

    def register_attn_weight_hook(self, hook: AttentionWeightHook) -> RemovableHandle:
        """Register an attention weight hook on the module.

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
        """Run registered attention weight hooks.

        A :class:`MultiheadAttention` implementation should call this method
        after computing attention weights in its :meth:`forward` method.

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


class InternalQKVProjection(ResettableProjection):
    def __init__(
        self,
        model_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    q_proj: Projection
    k_proj: Projection
    v_proj: Projection
    pos_embed: Optional[PositionalEmbedding]
    bias_k: Optional[Parameter]
    bias_v: Optional[Parameter]
    add_zero_attn: bool
    attn_fn: AttentionFunction
    attn_dropout_p: float
    scale_heads: bool
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
        scale_heads: bool = False,
        out_proj: Optional[Projection] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        :param scale_heads:
            If ``True``, Head Scaling is applied as described in
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456`
        :param out_proj:
            The projection to produce final attentions. If ``None``, a
            default projection will be used.
        """
        if model_dim is None:
            if q_proj is not None:
                model_dim = q_proj.inp_dim
            else:
                raise ValueError("`model_dim` must be specified.")

        super().__init__(num_heads, model_dim)

        if q_proj is None and k_proj is None and v_proj is None:
            q_proj = InternalQKVProjection(model_dim, device=device, dtype=dtype)
            k_proj = InternalQKVProjection(model_dim, device=device, dtype=dtype)
            v_proj = InternalQKVProjection(model_dim, device=device, dtype=dtype)
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
            if (head_dim := k_proj.out_dim // num_heads) != pos_embed.embed_dim:
                raise ValueError(
                    f"`embed_dim` of `pos_embed` ({pos_embed.embed_dim}) does not match the size of the header key dimension ({head_dim})."
                )

            self.pos_embed = pos_embed
        else:
            self.register_module("pos_embed", None)

        if add_bias_kv:
            bias_k_shp = (num_heads, 1, k_proj.out_dim // num_heads)
            bias_v_shp = (num_heads, 1, v_proj.out_dim // num_heads)

            # (H, 1, K_h)
            self.bias_k = Parameter(torch.empty(bias_k_shp, device=device, dtype=dtype))
            # (H, 1, V_h)
            self.bias_v = Parameter(torch.empty(bias_v_shp, device=device, dtype=dtype))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        self.add_zero_attn = add_zero_attn

        if attn_fn is None:
            self.attn_fn = default_scaled_dot_product_attention
        else:
            self.attn_fn = attn_fn

        self.attn_dropout_p = attn_dropout_p

        if scale_heads:
            self.scale_heads_proj = Parameter(
                torch.ones(num_heads, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("scale_heads_proj", None)

        if out_proj is None:
            self.out_proj = InternalOutProjection(
                v_proj.out_dim, model_dim, device=device, dtype=dtype
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
        """Reset the parameters of the module."""
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
        if padding_mask is not None:
            padding_mask = to_float_mask(padding_mask, dtype=keys.dtype)

        # (*, M) -> (N, S, K_proj)
        q = self._forward_proj(self.q_proj, x)

        if self.training or state_bag is None:
            # (*, K) -> (N, S_kv, K_proj)
            k = self._forward_proj(self.k_proj, keys)
            # (*, V) -> (N, S_kv, V_proj)
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
                    # (*, K) -> (N, S_kv, K_proj)
                    k = self._forward_proj(self.k_proj, keys)
                    # (*, V) -> (N, S_kv, V_proj)
                    v = self._forward_proj(self.v_proj, values)

                    state_bag.set_state(self, MultiheadAttentionState(k, v))
            else:
                # (*, K) -> (N, S_kv, K_proj)
                k = self._forward_proj(self.k_proj, keys)
                # (*, V) -> (N, S_kv, V_proj)
                v = self._forward_proj(self.v_proj, values)

                if state is not None:
                    k, v, padding_mask = state.append(k, v, padding_mask)
                else:
                    state_bag.set_state(
                        self, MultiheadAttentionState(k, v, padding_mask)
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
                attn_mask = F.pad(attn_mask, (0, mask_pad))

            if padding_mask is not None:
                # (N, S_kv) -> (N, S_kv + mask_pad)
                padding_mask = F.pad(padding_mask, (0, mask_pad))

        if padding_mask is not None:
            # (N, S_kv) -> (N, 1, 1, S_kv)
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            # (N, 1, 1, S_kv) -> (N, H, 1, S_kv)
            padding_mask = padding_mask.expand(-1, self.num_heads, -1, -1)

            if attn_mask is None:
                # (N, H, 1, S_kv)
                attn_mask = padding_mask
            else:
                # (N, H, 1, S_kv) + ([H,], S, S_kv) = (N, H, S, S_kv)
                attn_mask = padding_mask + attn_mask

            # (N, H, S, S_kv) -> (N x H, 1, S_kv)
            attn_mask = attn_mask.flatten(0, 1)

        needs_weights = len(self._attn_weight_hooks) > 0

        # attn:         (N x H, S, V_h)
        # attn_weights: (N x H, S, S_kv)
        attn, attn_weights = self.attn_fn(
            q, k, v, attn_mask, self.attn_dropout_p, needs_weights, self.training
        )

        if attn_weights is not None:
            self._run_attn_weight_hooks(attn_weights)

        # (N x H, S, V_h) -> (N, H, S, V_h)
        attn = attn.unflatten(0, (-1, self.num_heads))

        # (N, H, S, V_h) -> (N, S, H, V_h)
        attn = attn.permute(0, 2, 1, 3)

        if self.scale_heads_proj is not None:
            attn = torch.einsum("nshv,h->nshv", attn, self.scale_heads_proj)

        # (N, S, H, V_h) -> (N, S, V_proj)
        attn = attn.flatten(-2, -1)

        # (N, S, V_proj) -> (N, S, M)
        attn = self.out_proj(attn)

        if x.dim() == 2:
            # (1, S, M) -> (S, M)
            attn = attn.squeeze(0)

        return attn

    def _forward_proj(self, fn: Projection, x: Tensor) -> Tensor:
        x = fn(x)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        if self.add_zero_attn:
            s += ", add_zero_attn=True"

        return f"{s}, attn_fn={get_name(self.attn_fn)}, attn_dropout_p={self.attn_dropout_p}"
