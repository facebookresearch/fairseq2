# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, cast, final

import torch
import torch.nn as nn
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Module
from torch.nn.parameter import Parameter

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.transformer.ffn import FeedForwardNetwork
from fairseq2.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2.nn.transformer.norm_order import TransformerNormOrder


class TransformerDecoderLayer(Module, ABC):
    """Represents a Transformer decoder layer."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        enc_out: Optional[Tensor] = None,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param self_attn_mask:
            The float mask that will be added to the attention weights before
            computing the self attention. *Shape:* :math:`(S,S)`, where
            :math:`S` is the sequence length.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{src},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{src}` is the source sequence length, and :math:`M_{enc}`
            is the encoder model size.
        :param enc_padding_mask:
            The float padding mask of ``enc_out``. *Shape:* :math:`(N,S_{src})`,
            where :math:`N` is the batch size and :math:`S_{src}` is the source
            sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The decoded output of ``seqs``. *Shape:* Same as ``seqs``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    enc_dec_attn: Optional[MultiheadAttention]
    enc_dec_dropout: Optional[Dropout]
    enc_dec_attn_layer_norm: Optional[LayerNorm]
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456` for more
            information.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
        :param norm_order:
            The Layer Normalization order to use.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` modules for numerical stability.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        self_attn_layer_norm = LayerNorm(
            model_dim, norm_eps, device=device, dtype=dtype
        )

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = LayerNorm(
                model_dim, norm_eps, device=device, dtype=dtype
            )
        else:
            self.register_module("self_attn_norm", None)

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        if enc_dec_attn is None:
            self.register_module("enc_dec_attn", None)
            self.register_module("enc_dec_attn_layer_norm", None)
        else:
            if enc_dec_attn.model_dim != model_dim:
                raise ValueError(
                    f"`model_dim` of `enc_dec_attn` and `model_dim` of `self_attn` must be equal, but are {enc_dec_attn.model_dim} and {model_dim} instead."
                )

            enc_dec_attn_layer_norm = LayerNorm(
                model_dim, norm_eps, device=device, dtype=dtype
            )

            if norm_order != TransformerNormOrder.POST:
                self.enc_dec_attn_layer_norm = enc_dec_attn_layer_norm

            self.enc_dec_attn = enc_dec_attn

            if dropout_p > 0.0:
                self.enc_dec_attn_dropout = Dropout(dropout_p)
            else:
                self.register_module("enc_dec_attn_dropout", None)

            if norm_order == TransformerNormOrder.POST:
                self.enc_dec_attn_layer_norm = enc_dec_attn_layer_norm

        if ffn.model_dim != model_dim:
            raise ValueError(
                f"`model_dim` of `ffn` and `model_dim` of `self_attn` must be equal, but are {ffn.model_dim} and {model_dim} instead."
            )

        ffn_layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if scale_residual:
            self.residual_scale = Parameter(
                torch.empty((model_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        enc_out: Optional[Tensor] = None,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask, state_bag)

        seqs = self._forward_enc_dec_attn(
            seqs, padding_mask, enc_out, enc_padding_mask, state_bag
        )

        seqs = self._forward_ffn(seqs)

        return seqs

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            values=seqs,
            attn_mask=self_attn_mask,
            key_padding_mask=padding_mask,
            state_bag=state_bag,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_enc_dec_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        enc_out: Optional[Tensor],
        enc_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        if self.enc_dec_attn is None:
            if enc_out is not None:
                raise ValueError("`enc_out` must be `None` for decoder-only attention.")

            return seqs

        if enc_out is None:
            raise ValueError(
                "`enc_out` must not be `None` for encoder-decoder attention."
            )

        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.enc_dec_attn_layer_norm)(seqs)

        seqs = self.enc_dec_attn(
            seqs,
            padding_mask,
            keys=enc_out,
            values=enc_out,
            key_padding_mask=enc_padding_mask,
            state_bag=state_bag,
        )

        if self.enc_dec_attn_dropout is not None:
            seqs = self.enc_dec_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.enc_dec_attn_layer_norm)(seqs)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        if self.residual_scale is not None:
            residual = torch.mul(self.residual_scale, residual)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", norm_order={self.norm_order}"
