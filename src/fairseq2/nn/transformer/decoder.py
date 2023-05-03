# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Iterable, Optional, final

import torch
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import LayerNorm, Module

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.transformer.attention_mask import (
    AttentionMaskGenerator,
    CausalAttentionMaskGenerator,
)
from fairseq2.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2.nn.transformer.norm_order import TransformerNormOrder


class TransformerDecoder(Module, ABC):
    """Represents a Transformer decoder."""

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
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
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
        :param encoder_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{src},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{src}` is the source sequence length, and :math:`M_{enc}`
            is the encoder model size.
        :param encoder_padding_mask:
            The float padding mask of ``encoder_out``. *Shape:*
            :math:`(N,S_{src})`, where :math:`N` is the batch size and
            :math:`S_{src}` is the source sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The decoded output of ``seqs``. *Shape:* Same as ``seqs``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoder(TransformerDecoder):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn_mask_gen: AttentionMaskGenerator
    layers: ModuleList
    layer_norm: Optional[LayerNorm]

    def __init__(
        self,
        layers: Iterable[TransformerDecoderLayer],
        self_attn_mask_gen: Optional[AttentionMaskGenerator] = None,
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        :param self_attn_mask_gen:
            The attention mask generator. If ``None``, an instance of
            :class:`CausalAttentionMaskGenerator` will be used.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the decoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param norm_order:
            The Layer Normalization order to use.
        :param norm_eps:
            The epsilon value to add to the denominator of the
            :class:`~torch.nn.LayerNorm` module for numerical stability.
        """
        layer_list = ModuleList(layers, layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        for idx, layer in enumerate(layers):
            if layer.model_dim != model_dim:
                raise ValueError(
                    f"`model_dim` of the decoder layer 0 and `model_dim` of the decoder layer {idx} must be equal, but are {model_dim} and {layer.model_dim} instead."
                )

        super().__init__(model_dim)

        if self_attn_mask_gen is None:
            self.self_attn_mask_gen = CausalAttentionMaskGenerator()
        else:
            self.self_attn_mask_gen = self_attn_mask_gen

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = LayerNorm(model_dim, norm_eps, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor] = None,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        if self.training or state_bag is None:
            self_attn_mask = self.self_attn_mask_gen(seqs)
        else:
            self_attn_mask = None

        for layer in self.layers.drop_iter():
            seqs = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_out,
                encoder_padding_mask,
                state_bag,
            )

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        mask_gen_name = getattr(
            self.self_attn_mask_gen, "__name__", repr(self.self_attn_mask_gen)
        )

        return s + f", self_attn_mask_gen={mask_gen_name}"
