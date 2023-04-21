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
from fairseq2.nn.utils.mask import to_float_mask


class TransformerDecoder(Module, ABC):
    """Represents a Transformer decoder."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model (i.e. inputs and outputs).
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        embeds: Tensor,
        padding_mask: Optional[Tensor] = None,
        enc_out: Optional[Tensor] = None,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        """
        :param embeds:
            The embeddings to decode. *Shape:* :math:`(N,S,M)`, or :math:`(S,M)`
            when unbatched, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.
        :param padding_mask:
            The boolean or float padding mask indicating which key positions to
            ignore for the purpose of self attention. *Shape:* :math:`(N,S)`, or
            :math:`(S)` when unbatched, where :math:`N` is the batch size and
            :math:`S` is the sequence length.
        :param enc_out:
            The encoder output for the encoder-decoder attention. *Shape:*
            :math:`(N,S_{src},M_{enc})`, or :math:`(S_{src},M_{enc})` when
            unbatched, where :math:`N` is the batch size, :math:`S_{src}` is the
            source sequence length, and :math:`M_{enc}` is the encoder model
            size.
        :param enc_padding_mask:
            The boolean or float padding mask indicating which key positions to
            ignore for the purpose of encoder-decoder attention. *Shape:*
            :math:`(N,S_{src})`, or :math:`(S_{src})` when unbatched, where
            :math:`N` is the batch size and :math:`S_{src}` is the source
            sequence length.
        :param state_bag:
            The state bag to use during an incremental evaluation.

        :returns:
            The decoded output of ``embeds``. *Shape:* Same as ``embeds``.

        .. note::
            For a boolean padding mask, a ``True`` indicates that the
            corresponding key position is not allowed to attend. For a float
            padding mask, the mask values will be added to the attention
            weights.
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
        embeds: Tensor,
        padding_mask: Optional[Tensor] = None,
        enc_out: Optional[Tensor] = None,
        enc_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tensor:
        if padding_mask is not None:
            padding_mask = to_float_mask(padding_mask, dtype=embeds.dtype)

        if self.training or state_bag is None:
            self_attn_mask = self.self_attn_mask_gen(embeds)
        else:
            self_attn_mask = None

        x = embeds

        for layer in self.layers.drop_iter():
            x = layer(
                x, padding_mask, self_attn_mask, enc_out, enc_padding_mask, state_bag
            )

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        mask_gen_name = getattr(
            self.self_attn_mask_gen, "__name__", repr(self.self_attn_mask_gen)
        )

        return s + f", self_attn_mask_gen={mask_gen_name}"
