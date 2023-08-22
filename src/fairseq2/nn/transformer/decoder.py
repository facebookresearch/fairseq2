# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Protocol, Tuple, final

from torch import Tensor
from torch.nn import Module

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.transformer.attention_mask import (
    AttentionMaskGenerator,
    CausalAttentionMaskGenerator,
)
from fairseq2.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.nn.utils.module import check_model_dim
from fairseq2.typing import DataType, Device, finaloverride


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
        state_bag: Optional[IncrementalStateBag] = None,
        layer_output_hook: Optional["DecoderLayerOutputHook"] = None,
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
        :param state_bag:
            The state bag to use for incremental evaluation.
        :param layer_output_hook:
            If not ``None``, it will be called with the output of each layer in
            the decoder stack.

        :returns:
            - The decoder output. *Shape:* Same as ``seqs``.
            - The float padding mask of the decoder output. *Shape:* Same as
              ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


class DecoderLayerOutputHook(Protocol):
    """Represents a hook to pass to :meth:`~TransformerDecoder.forward`."""

    def __call__(
        self,
        layer_idx: int,
        layer_output: Tensor,
        layer_padding_mask: Optional[Tensor],
        num_layers: int,
    ) -> None:
        """
        :param layer_idx:
            The index of the layer in the decoder stack.
        :param layer_output:
            The decoded output of the layer.
        :param layer_padding_mask:
            The padding mask of `layer_output`.
        :param num_layers:
            The number of layers in the decoder stack.
        """


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
        layer_norm_fn: Optional[LayerNormFactory] = None,
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
        :param layer_norm_fn:
            The factory to use to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers, layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_fn is None:
            layer_norm_fn = create_default_layer_norm

        if self_attn_mask_gen is None:
            self.self_attn_mask_gen = CausalAttentionMaskGenerator()
        else:
            self.self_attn_mask_gen = self_attn_mask_gen

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = layer_norm_fn(model_dim, device, dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

        check_model_dim(self)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
        layer_output_hook: Optional[DecoderLayerOutputHook] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if layer_output_hook is not None and self.layers.drop_p > 0.0:
            raise ValueError("`layer_hook` must be `None` when LayerDrop is enabled.")

        num_layers = len(self.layers)

        if seqs.size(1) > 1:
            self_attn_mask = self.self_attn_mask_gen(seqs)
        else:
            self_attn_mask = None

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_output,
                encoder_padding_mask,
                state_bag,
            )

            if layer_output_hook is not None:
                layer_output_hook(layer_idx, seqs, padding_mask, num_layers)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        mask_gen_name = getattr(
            self.self_attn_mask_gen, "__name__", repr(self.self_attn_mask_gen)
        )

        return s + f", norm_order={self.norm_order}, self_attn_mask_gen={mask_gen_name}"
