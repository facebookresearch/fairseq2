# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Type, final

from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.nn.transformer.attention_mask import (
    AttentionMaskGenerator,
    CausalAttentionMaskGenerator,
)
from fairseq2.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.typing import DataType, Device


class TransformerDecoder(Module, ABC):
    """Represents a Transformer decoder."""

    model_dim: int
    layers: ModuleList

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
        padding_mask: Optional[Tensor],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        return_hidden: Optional[int] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences to decode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and
            :math:`M_{enc}` is the dimensionality of the encoder.
        :param encoder_padding_mask:
            The float padding mask of ``encoder_out``. *Shape:*
            :math:`(N,S_{enc})`, where :math:`N` is the batch size and
            :math:`S_{enc}` is the encoder output sequence length.
        :param return_hidden:
            If not ``None``, specifies the index of the decoder layer whose
            output should be returned along with the decoder output.
        :param state_bag:
            The state bag to use for incremental evaluation.

        :returns:
            - The decoder output. *Shape:* Same as ``seqs``.
            - The output of the decoder layer specified by ``return_hidden``.
              *Shape:* Same as ``seqs``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoder(TransformerDecoder):
    """Represents a Transformer decoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn_mask_gen: AttentionMaskGenerator
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[TransformerDecoderLayer],
        self_attn_mask_gen: Optional[AttentionMaskGenerator] = None,
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_kls: Optional[Type[LayerNorm]] = None,
        norm_eps: float = 1e-5,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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
        :param layer_norm_kls:
            The type of Layer Normalization to use.
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
            if layer_norm_kls is None:
                layer_norm_kls = StandardLayerNorm

            self.layer_norm = layer_norm_kls(
                model_dim, norm_eps, device=device, dtype=dtype
            )
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        return_hidden: Optional[int] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if return_hidden is not None:
            if self.layers.drop_p > 0.0:
                raise ValueError(
                    "`return_hidden` must be `None` when LayerDrop is enabled."
                )

            if return_hidden < 0:
                return_hidden = len(self.layers) + return_hidden

        layer_output = None

        if self.training or state_bag is None:
            self_attn_mask = self.self_attn_mask_gen(seqs)
        else:
            self_attn_mask = None

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_output,
                encoder_padding_mask,
                state_bag,
            )

            if layer_idx == return_hidden:
                layer_output = seqs

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, layer_output

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        mask_gen_name = getattr(
            self.self_attn_mask_gen, "__name__", repr(self.self_attn_mask_gen)
        )

        return s + f", norm_order={self.norm_order}, self_attn_mask_gen={mask_gen_name}"
