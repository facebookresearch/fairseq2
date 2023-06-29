# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, final

from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.transformer.encoder_layer import TransformerEncoderLayer
from fairseq2.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2.nn.transformer.norm_order import TransformerNormOrder
from fairseq2.typing import DataType, Device


class TransformerEncoder(Module, ABC):
    """Represents a Transformer encoder."""

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
        return_hidden: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences to encode. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param return_hidden:
            If not ``None``, specifies the index of the encoder layer whose
            output should be returned along with the encoder output.

        :returns:
            - The encoder output. *Shape:* Same as ``seqs``.
            - The output of the encoder layer specified by ``return_hidden``.
              *Shape:* Same as ``seqs``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerEncoder(TransformerEncoder):
    """Represents a Transformer encoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[TransformerEncoderLayer],
        layer_drop_p: float = 0.0,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_fn: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The encoder layers.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the encoder layers as
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

        for idx, layer in enumerate(layers):
            if layer.model_dim != model_dim:
                raise ValueError(
                    f"`model_dim` of the encoder layer 0 and `model_dim` of the encoder layer {idx} must be equal, but are {model_dim} and {layer.model_dim} instead."
                )

        super().__init__(model_dim)

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            if layer_norm_fn is None:
                layer_norm_fn = create_default_layer_norm

            self.layer_norm = layer_norm_fn(model_dim, device, dtype)
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        return_hidden: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if return_hidden is not None:
            if self.layers.drop_p > 0.0:
                raise ValueError(
                    "`return_hidden` must be `None` when LayerDrop is enabled."
                )

            if return_hidden < 0:
                return_hidden = len(self.layers) + return_hidden

        layer_output = None

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs = layer(seqs, padding_mask)

            if layer_idx == return_hidden:
                layer_output = seqs

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, layer_output

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", norm_order={self.norm_order}"
