# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, cast, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import final as finaloverride
from torch import Tensor
from torch import dtype as DataType
from torch.nn import LayerNorm, Module, Parameter

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.transformer.ffn import FeedForwardNetwork
from fairseq2.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2.nn.transformer.norm_order import TransformerNormOrder


class TransformerDecoderLayer(Module, ABC):
    """Represents a Transformer decoder layer."""

    model_dim: int
    """The dimensionality of the model (i.e. inputs and outputs)."""

    batch_first: bool
    """If ``True``, the first dimension of batched inputs and outputs represents
    the batch; otherwise, the sequence."""

    def __init__(self, model_dim: int, batch_first: bool) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        :param batch_first:
            If ``True``, the first dimension of batched inputs and outputs
            represents the batch; otherwise, the sequence.
        """
        super().__init__()

        self.model_dim = model_dim

        self.batch_first = batch_first

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        enc_out: Optional[Tensor] = None,
        enc_attn_padding_mask: Optional[Tensor] = None,
        incremental_state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param x:
            The input to process. *Shape:* :math:`(T,M)` when unbatched,
            :math:`(N,T,M)` when :attr:`batch_first` is ``True``, or
            :math:`(T,N,M)` when :attr:`batch_first` is ``False``, where
            :math:`N` is the batch size, :math:`T` is the target sequence
            length, and :math:`M` is the model size.
        :param self_attn_mask:
            The float mask that will be added to the attention weights before
            computing the self attention. *Shape:* :math:`(T,T)`, where
            :math:`T` is the target sequence length.
        :param self_attn_padding_mask:
            The boolean or float key padding mask indicating which key positions
            to ignore for the purpose of self attention. *Shape:* :math:`(T)`
            when unbatched, :math:`(N,T)` when :attr:`batch_first` is ``True``,
            or :math:`(T,N)` when :attr:`batch_first` is ``False``, where
            :math:`N` is the batch size and :math:`T` is the target sequence
            length.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(S,M_{enc})` when unbatched, :math:`(N,S,M_{enc})` when
            :attr:`batch_first` is ``True``, or :math:`(S,N,M_{enc})` when
            :attr:`batch_first` is ``False``, where :math:`N` is the batch size,
            :math:`S` is the source sequence length, and :math:`M_{enc}` is the
            encoder model size.
        :param enc_attn_padding_mask:
            The boolean or float key padding mask indicating which key positions
            to ignore for the purpose of encoder-decoder attention. *Shape:*
            :math:`(S)` when unbatched, :math:`(N,S)` when :attr:`batch_first`
            is ``True``, or :math:`(S,N)` when :attr:`batch_first` is ``False``,
            where :math:`N` is the batch size and :math:`S` is the source
            sequence length.
        :param incremental_state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The output. *Shape:* Same as ``x``.

        .. note::
            For a boolean key padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend. For a float key
            padding mask, the mask values will be added to the attention
            weights.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`DBLP:journals/corr/VaswaniSPUJGKP17`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_layer_norm: LayerNorm
    enc_dec_attn: Optional[MultiheadAttention]
    enc_dec_attn_layer_norm: Optional[LayerNorm]
    ffn: FeedForwardNetwork
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    dropout_p: float
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        enc_dec_attn: Optional[MultiheadAttention],
        ffn: FeedForwardNetwork,
        scale_residual: bool = False,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-5,
        device: Any = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param enc_dec_attn:
            The encoder-decoder attention layer.
        :param ffn:
            The feed-forward network.
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network. See
            :cite:t:`DBLP:journals/corr/abs-2110-09456` for more information.
        :param dropout_p:
            The dropout probability on the outputs of the attention layers and
            the feed-forward network.
        :param norm_order:
            The Layer Normalization order to use.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` modules for numerical stability.
        """
        fct_kwargs: Dict = {"device": device, "dtype": dtype}

        model_dim, batch_first = self_attn.model_dim, self_attn.batch_first

        super().__init__(model_dim, batch_first)

        self_attn_layer_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)
        else:
            self.register_module("self_attn_norm", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        if enc_dec_attn is None:
            self.register_module("enc_dec_attn", None)
            self.register_module("enc_dec_attn_layer_norm", None)
        else:
            if enc_dec_attn.model_dim != model_dim:
                raise ValueError(
                    f"`model_dim` of `enc_dec_attn` ({enc_dec_attn.model_dim}) does not match `model_dim` ({model_dim})."
                )

            if enc_dec_attn.batch_first != batch_first:
                raise ValueError(
                    f"`batch_first` of `enc_dec_attn` ({enc_dec_attn.batch_first}) does not match `batch_first` ({batch_first})."
                )

            enc_dec_attn_layer_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)

            if norm_order != TransformerNormOrder.POST:
                self.enc_dec_attn_layer_norm = enc_dec_attn_layer_norm

            self.enc_dec_attn = enc_dec_attn

            if norm_order == TransformerNormOrder.POST:
                self.enc_dec_attn_layer_norm = enc_dec_attn_layer_norm

        if ffn.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `ffn` ({ffn.model_dim}) does not match `model_dim` ({model_dim})."
            )

        ffn_layer_norm = LayerNorm(model_dim, norm_eps, **fct_kwargs)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if scale_residual:
            self.residual_scale = Parameter(torch.empty(model_dim, **fct_kwargs))
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.dropout_p = dropout_p

        self.norm_order = norm_order

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
        self,
        x: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        enc_out: Optional[Tensor] = None,
        enc_attn_padding_mask: Optional[Tensor] = None,
        incremental_state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        x = self._forward_self_attn(
            x,
            self_attn_mask,
            self_attn_padding_mask,
            incremental_state_bag,
        )

        x = self._forward_enc_dec_attn(
            x,
            enc_out,
            enc_attn_padding_mask,
            incremental_state_bag,
        )

        x = self._forward_ffn(x)

        return x

    def _forward_self_attn(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        attn_padding_mask: Optional[Tensor],
        incremental_state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        residual = x

        if self.norm_order != TransformerNormOrder.POST:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(
            x,
            keys=x,
            values=x,
            attn_mask=attn_mask,
            padding_mask=attn_padding_mask,
            incremental_state_bag=incremental_state_bag,
        )

        if self.self_attn_norm is not None:
            x = self.self_attn_norm(x)

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.self_attn_layer_norm(x)

        return x

    def _forward_enc_dec_attn(
        self,
        x: Tensor,
        enc_out: Optional[Tensor],
        attn_padding_mask: Optional[Tensor],
        incremental_state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        if self.enc_dec_attn is None:
            return x

        if enc_out is None:
            raise ValueError(
                "`enc_out` must not be `None` for encoder-decoder attention."
            )

        residual = x

        if self.norm_order != TransformerNormOrder.POST:
            x = cast(LayerNorm, self.enc_dec_attn_layer_norm)(x)

        x = self.enc_dec_attn(
            x,
            keys=enc_out,
            values=enc_out,
            padding_mask=attn_padding_mask,
            incremental_state_bag=incremental_state_bag,
        )

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = cast(LayerNorm, self.enc_dec_attn_layer_norm)(x)

        return x

    def _forward_ffn(self, x: Tensor) -> Tensor:
        residual = x

        if self.norm_order != TransformerNormOrder.POST:
            x = self.ffn_layer_norm(x)

        x = self.ffn(x)

        if self.dropout_p > 0.0:
            x = F.dropout(x, self.dropout_p, self.training)

        if self.residual_scale is not None:
            residual = torch.mul(self.residual_scale, residual)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.ffn_layer_norm(x)

        return x
